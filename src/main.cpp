#include <iostream>
#include <vector>

#include "dg_solver.hpp"
#include "params.hpp"
#include "quadrature_sn.hpp"
#include "callaway_rta.hpp"
#include "equilibrium_gray.hpp"
#include "dg/transport.hpp"
#include "normal_closure.hpp"
#include "gray_sn_steady.hpp"
#include "postprocess.hpp"
#include <mfem.hpp>
#include "cli.hpp"

int main(int argc, char** argv) {
    std::cout << "START\n";
    dg::CLIConfig cfg;
    dg::ParseCLI(argc, argv, cfg);
    std::cout << "PARSED\n";

    // std::cout << "DG4PBTE demo: basic build and clangd test\n";

    // std::vector<double> values = {1.0, 2.0, 3.0, 4.0};
    // double l2 = computeL2Norm(values);

    // std::cout << "L2 norm of {1,2,3,4} = " << l2 << "\n";
    
    // // Minimal MFEM sanity check: build a tiny mesh, finite element space, and project a constant
    // try {
    //     mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D(4);
    //     mfem::H1_FECollection fec(1, mesh.Dimension());
    //     mfem::FiniteElementSpace fes(&mesh, &fec);
    //     mfem::GridFunction u(&fes);
    //     mfem::ConstantCoefficient one(1.0);
    //     u.ProjectCoefficient(one);
    //     std::cout << "MFEM OK: dim=" << mesh.Dimension()
    //               << ", elements=" << mesh.GetNE()
    //               << ", dofs=" << fes.GetNDofs()
    //               << ", ||u||_L2=" << u.Norml2() << "\n";
    // } catch (const std::exception &e) {
    //     std::cerr << "MFEM test failed: " << e.what() << "\n";
    //     return 1;
    // }

    // Minimal Callaway double-relaxation operator sanity check (gray).
    dg::GrayCallawayParams params = dg::MakeParams(cfg);

    const auto dirs = dg::MakeSN(cfg);
    dg::CallawayRTA coll(params);

    std::vector<double> g(dirs.omega.size(), 1.0);
    std::vector<double> rhs;
    // Define simple gray equilibria: BE(T) and drifted BE(T,u)
    const double T = cfg.T;
    const std::array<double,3> u = {cfg.ux, cfg.uy, cfg.uz};
    dg::GrayEquilibrium geq;
    geq.referenceTemperature = cfg.T;
    geq.driftCoefficient = 0.5; // scale of drift contribution
    dg::EquilibriumFields eq;
    eq.resistive = [T, geq](const std::array<double,3>&){ return geq.resistiveBE(T); };
    eq.normal    = [T, u, geq](const std::array<double,3>& om){ return geq.normalDriftedBE(T, om, u); };
    coll.apply(g, eq, dirs, /*L_char=*/cfg.Lchar, rhs);
    if (!rhs.empty()) {
        std::cout << "Callaway RTA OK: C[g]_0 = " << rhs[0] << "\n";
    }

    // Demonstrate normal-scattering drift closure (linearized) from current g.
    {
        const double base = geq.resistiveBE(T);
        const double alpha = geq.driftCoefficient;
        const auto u_closure = dg::computeNormalDriftLinearized(dirs, g, base, alpha);
        std::cout << "Normal drift closure u = (" << u_closure[0] << ", "
                  << u_closure[1] << ", " << u_closure[2] << ")\n";
    }

    // Source iteration: solve all directions with current equilibria, then
    // update macroscopic fields from angular average until convergence.
    {
        mfem::Mesh mesh = (cfg.dim == 2)
            ? mfem::Mesh::MakeCartesian2D(cfg.nx, cfg.ny, mfem::Element::QUADRILATERAL, true, cfg.Lx, cfg.Ly)
            : mfem::Mesh::MakeCartesian3D(cfg.nx, cfg.ny, cfg.nz, mfem::Element::HEXAHEDRON, cfg.Lx, cfg.Ly, cfg.Lz);
        const int dim = mesh.Dimension();
        mfem::L2_FECollection fec(cfg.p, dim);
        mfem::FiniteElementSpace fes(&mesh, &fec);

        std::vector<std::unique_ptr<mfem::GridFunction>> sol(dirs.omega.size());
        mfem::GridFunction ang_avg(&fes), ang_prev(&fes), one_gf(&fes);
        ang_avg = 0.0;
        ang_prev = 0.0;
        mfem::ConstantCoefficient one_c(1.0);
        one_gf.ProjectCoefficient(one_c);
        mfem::LinearForm lf(&fes);
        lf.AddDomainIntegrator(new mfem::DomainLFIntegrator(one_c));
        lf.Assemble();
        const double domain_measure = lf(one_gf);

        double Tref = cfg.T;
        mfem::GridFunction Tfield(&fes);
        {
            mfem::ConstantCoefficient T0(Tref);
            Tfield.ProjectCoefficient(T0);
        }
        dg::GrayEquilibrium geq_it = geq; // copy, carries reference scaling
        geq_it.referenceTemperature = Tref;

        // element-average temperature vectors for residual
        mfem::Vector Tv_prev(mesh.GetNE());
        mfem::Vector Tv_cur(mesh.GetNE());
        Tv_prev = 0.0;
        Tv_cur = 0.0;

        auto compute_element_averages = [&](const mfem::GridFunction& gf, mfem::Vector& out){
            const int ne = mesh.GetNE();
            out.SetSize(ne);
            for (int e = 0; e < ne; ++e) {
                mfem::ElementTransformation *Tr_el = mesh.GetElementTransformation(e);
                const mfem::FiniteElement *fe = fes.GetFE(e);
                const int order = 2*fe->GetOrder() + 2;
                const mfem::IntegrationRule &ir = mfem::IntRules.Get(fe->GetGeomType(), order);
                double num = 0.0, den = 0.0;
                for (int i = 0; i < ir.GetNPoints(); ++i) {
                    const mfem::IntegrationPoint &ip = ir.IntPoint(i);
                    Tr_el->SetIntPoint(&ip);
                    const double w = ip.weight * Tr_el->Weight();
                    double val = gf.GetValue(e, ip);
                    num += val * w;
                    den += w;
                }
                out[e] = (den != 0.0) ? (num/den) : 0.0;
            }
        };

        for (int it = 0; it < cfg.maxIters; ++it)
        {
            ang_prev = ang_avg;
            ang_avg = 0.0;

            // No attribute map for now; use coordinate-based hot wall: x=0 hot, others cold.

            for (size_t i = 0; i < dirs.omega.size(); ++i)
            {
                // For inflow RHS we still use eq from earlier (resistive(Tref)), it's fine for now.
                auto sys = dg::buildDirectionSystem(fes, dirs, /*idir=*/(int)i, params, cfg.Lchar,
                                                    eq, geq_it, &Tfield, u);
                // Add boundary penalty terms before assembling
                // (block below may no-op if mesh has no attributes or bc disabled)
                // Build boundary markers for hot/cold and add integrators with fallback lambdas.
                if (cfg.useIsoBC && mesh.bdr_attributes.Size())
                {
                    // Build a per-attribute boundary map: attribute -> Tw (mapped to BE(Tw))
                    auto bmap = std::make_unique<dg::BoundaryConditionMap>();
                    for (int k = 0; k < mesh.bdr_attributes.Size(); ++k)
                    {
                        int attr = mesh.bdr_attributes[k];
                        dg::BoundaryData bd;
                        bd.type = dg::BoundaryType::Isothermal;
                        const bool is_hot = (attr == cfg.hotAttr);
                        const double Tw = is_hot ? cfg.Thot : cfg.Tcold;
                        bd.wallTemperature = geq_it.resistiveBE(Tw); // store as g_in value
                        (*bmap)[attr] = bd;
                    }
                    // Keep map alive in system, then pass pointer to integrator.
                    const dg::BoundaryConditionMap* bmap_ptr = bmap.get();
                    sys.owned_bmaps.push_back(std::move(bmap));
                    sys.bform->AddBdrFaceIntegrator(
                        new dg::InflowBoundaryRHS(dim, params.groupVelocity, dirs.omega[i],
                                                  /*bdr_map*/bmap_ptr,
                                                  /*fallback*/nullptr));
                }
                // Now assemble forms
                sys.Aform->Assemble();
                sys.Aform->Finalize();
                if (cfg.useIsoBC && mesh.bdr_attributes.Size())
                {
                    // Assemble while boundary markers are still in scope
                    sys.bform->Assemble();
                }
                else
                {
                    sys.bform->Assemble();
                }
                if (i == 0 && it == 0) {
                    auto &A0 = sys.Aform->SpMat();
                    std::cout << "DG Steady (dir0) assembled: size=" << A0.Height()
                              << " nnz=" << A0.NumNonZeroElems() << "\n";
                }
                dg::solveDirectionSystem(sys);
                sol[i] = std::move(sys.solution);
                ang_avg.Add(dirs.weight[i], *sol[i]);
            }

            // Update temperature field pointwise: T := (1-relax) T + relax * Tref * <g>
            mfem::GridFunction Tnew(&fes);
            Tnew = Tfield;
            Tnew *= (1.0 - cfg.relax);
            mfem::GridFunction tmp = ang_avg; // copy
            tmp *= (Tref * cfg.relax);
            Tnew += tmp;

            // Compute element-average vectors for residual
            compute_element_averages(Tfield, Tv_prev);
            compute_element_averages(Tnew, Tv_cur);

            // Swap into Tfield
            Tfield = Tnew;

            // Residual: ||Tv - Tv_old|| / ||Tv_old||
            mfem::Vector diffv = Tv_cur;
            diffv -= Tv_prev;
            const double num = diffv.Norml2();
            const double den = std::max(Tv_prev.Norml2(), 1e-16);
            const double rel = num / den;
            std::cout << "iter " << it+1 << ", residual = " << rel << "\n";
            if (rel < cfg.rtol) { break; }
        }

        std::cout << "Solved all " << dirs.omega.size() << " directions; "
                  << "||<g>_ang||_L2 = " << ang_avg.Norml2() << "\n";

        if (cfg.saveVTK) {
            // Heat flux components (gray, simple): q_c ≈ v_g Σ_i w_i ω_c,i g_i
            mfem::GridFunction qx(&fes), qy(&fes), qz(&fes);
            qx = 0.0; qy = 0.0; qz = 0.0;
            for (size_t i = 0; i < dirs.omega.size(); ++i) {
                const double factor = params.groupVelocity * dirs.weight[i];
                qx.Add(factor * dirs.omega[i][0], *sol[i]);
                if (dim > 1) qy.Add(factor * dirs.omega[i][1], *sol[i]);
                if (dim > 2) qz.Add(factor * dirs.omega[i][2], *sol[i]);
            }
            mfem::GridFunction *qxp = &qx;
            mfem::GridFunction *qyp = (dim > 1 ? &qy : nullptr);
            mfem::GridFunction *qzp = (dim > 2 ? &qz : nullptr);
            dg::SaveVTK(cfg.outPrefix, mesh, Tfield, &ang_avg, qxp, qyp, qzp, cfg.p, /*high_order=*/true);
            std::cout << "Saved VTK to prefix '" << cfg.outPrefix << "' (temperature, g_avg)\n";
        }
    }
    return 0;
}


