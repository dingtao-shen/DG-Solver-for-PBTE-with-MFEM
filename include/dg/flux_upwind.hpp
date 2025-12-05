#pragma once

#include <array>
#include <memory>
#include <functional>
#include <mfem.hpp>
#include "boundary.hpp"

namespace dg {

// Minimal scaffold for upwind numerical flux integrators.
// Currently assembles zero contributions to keep compilation and wiring clean.
// We'll fill actual upwind terms next.
class UpwindFaceIntegrator : public mfem::BilinearFormIntegrator {
public:
    UpwindFaceIntegrator(int dim,
                         double vg,
                         const std::array<double,3>& omega,
                         const BoundaryConditionMap* bdr_map = nullptr)
        : dim_(dim), vg_(vg), omega_(omega), bdr_map_(bdr_map) {}

    void AssembleFaceMatrix(const mfem::FiniteElement &el1,
                            const mfem::FiniteElement &el2,
                            mfem::FaceElementTransformations &Tr,
                            mfem::DenseMatrix &elmat) override
    {
        const int dof1 = el1.GetDof();
        const int dof2 = el2.GetDof();
        elmat.SetSize(dof1 + dof2);
        elmat = 0.0;

        const mfem::IntegrationRule *ir = IntRule;
        if (ir == nullptr)
        {
            int order = std::max(el1.GetOrder(), el2.GetOrder())*2 + 2;
            ir = &mfem::IntRules.Get(Tr.GetGeometryType(), order);
        }

        mfem::Vector shape1(dof1), shape2(dof2), normal(dim_), xphys;
        for (int i = 0; i < ir->GetNPoints(); i++)
        {
            const mfem::IntegrationPoint &ip = ir->IntPoint(i);
            Tr.SetAllIntPoints(&ip);

            // Face geometry and unit normal
            const mfem::DenseMatrix &J = Tr.Face->Jacobian();
            mfem::CalcOrtho(J, normal);
            double dS = normal.Norml2();
            if (dS > 0.0) { normal /= dS; }
            const double wgt = dS * ip.weight;

            // a·n
            const double an = vg_ * (omega_[0]*normal[0]
                + (dim_ > 1 ? omega_[1]*normal[1] : 0.0)
                + (dim_ > 2 ? omega_[2]*normal[2] : 0.0));
            const double an_pos = std::max(an, 0.0);
            const double an_neg = std::min(an, 0.0); // negative or zero

            // Shapes on both sides
            const mfem::IntegrationPoint &eip1 = Tr.GetElement1IntPoint();
            const mfem::IntegrationPoint &eip2 = Tr.GetElement2IntPoint();
            el1.CalcShape(eip1, shape1);
            el2.CalcShape(eip2, shape2);

            // Upwind contributions:
            // For test on elem1 (rows 0..dof1-1):
            //   if an>0: uses u1 -> block(1,1) += an * phi1_j * phi1_k
            //   else   : uses u2 -> block(1,2) += an * phi1_j * psi2_k
            if (an_pos > 0.0)
            {
                for (int j = 0; j < dof1; j++)
                    for (int k = 0; k < dof1; k++)
                        elmat(j, k) += an_pos * wgt * shape1[j] * shape1[k];
            }
            else if (an_neg < 0.0)
            {
                for (int j = 0; j < dof1; j++)
                    for (int k = 0; k < dof2; k++)
                        elmat(j, dof1 + k) += an_neg * wgt * shape1[j] * shape2[k];
            }

            // For test on elem2 (rows dof1..dof1+dof2-1) with outward normal -n:
            // an2 = a·(-n) = -an; if an<0 => an2>0 uses u2; else uses u1.
            const double an2_pos = std::max(-an, 0.0);
            const double an2_neg = std::min(-an, 0.0); // negative or zero
            if (an2_pos > 0.0) // i.e., an<0, upwind=u2
            {
                for (int j = 0; j < dof2; j++)
                    for (int k = 0; k < dof2; k++)
                        elmat(dof1 + j, dof1 + k) += an2_pos * wgt * shape2[j] * shape2[k];
            }
            else if (an2_neg < 0.0) // i.e., an>0, upwind=u1
            {
                for (int j = 0; j < dof2; j++)
                    for (int k = 0; k < dof1; k++)
                        elmat(dof1 + j, k) += an2_neg * wgt * shape2[j] * shape1[k];
            }
        }
    }

    void AssembleBdrFaceMatrix(const mfem::FiniteElement &el,
                               mfem::FaceElementTransformations &Tr,
                               mfem::DenseMatrix &elmat)
    {
        const int dof = el.GetDof();
        elmat.SetSize(dof);
        elmat = 0.0;

        const mfem::IntegrationRule *ir = IntRule;
        if (ir == nullptr)
        {
            int order = 2 * el.GetOrder() + 2;
            // Use face geometry for boundary integration
            MFEM_VERIFY(Tr.Face != nullptr, "Boundary face transformation is null");
            ir = &mfem::IntRules.Get(Tr.Face->GetGeometryType(), order);
        }

        mfem::Vector shape(dof), normal(dim_);
        for (int i = 0; i < ir->GetNPoints(); i++)
        {
            const mfem::IntegrationPoint &ip = ir->IntPoint(i);
            Tr.SetAllIntPoints(&ip);
            const mfem::DenseMatrix &J = Tr.Face->Jacobian();
            mfem::CalcOrtho(J, normal);
            double dS = normal.Norml2();
            if (dS > 0.0) { normal /= dS; }
            const double wgt = dS * ip.weight;
            const double an = vg_ * (omega_[0]*normal[0]
                + (dim_ > 1 ? omega_[1]*normal[1] : 0.0)
                + (dim_ > 2 ? omega_[2]*normal[2] : 0.0));
            if (an <= 0.0) { continue; } // inflow handled by RHS, no matrix

            const mfem::IntegrationPoint &eip = Tr.GetElement1IntPoint();
            el.CalcShape(eip, shape);
            for (int j = 0; j < dof; j++)
                for (int k = 0; k < dof; k++)
                    elmat(j, k) += an * wgt * shape[j] * shape[k];
        }
    }

private:
    int dim_;
    double vg_;
    std::array<double,3> omega_;
    const BoundaryConditionMap* bdr_map_; // optional, for boundary types
};

// RHS integrator for inflow Dirichlet-type boundary (injects g_in).
// Placeholder: currently zero to keep scaffold compiling.
class InflowBoundaryRHS : public mfem::LinearFormIntegrator {
public:
    using InflowValue = std::function<double(const std::array<double,3>&,
                                             const mfem::Vector& x)>;
    InflowBoundaryRHS(int dim,
                      double vg,
                      const std::array<double,3>& omega,
                      const BoundaryConditionMap* bdr_map,
                      InflowValue inflow_value)
        : dim_(dim), vg_(vg), omega_(omega), bdr_map_(bdr_map),
          inflow_value_(std::move(inflow_value)) {}

    void AssembleRHSElementVect(const mfem::FiniteElement &el,
                                mfem::FaceElementTransformations &Tr,
                                mfem::Vector &elvect) override
    {
        // Boundary face quadrature rule
        const mfem::IntegrationRule *ir = IntRule;
        if (ir == nullptr)
        {
            int order = 2 * el.GetOrder() + 2;
            MFEM_VERIFY(Tr.Face != nullptr, "Boundary face transformation is null");
            ir = &mfem::IntRules.Get(Tr.Face->GetGeometryType(), order);
        }
        elvect.SetSize(el.GetDof());
        elvect = 0.0;

        mfem::Vector shape(el.GetDof());
        mfem::Vector normal(dim_);
        mfem::Vector xphys;

        for (int i = 0; i < ir->GetNPoints(); i++)
        {
            const mfem::IntegrationPoint &ip = ir->IntPoint(i);
            // Configure all reference points for face and element 1
            Tr.SetAllIntPoints(&ip);
            const mfem::IntegrationPoint &eip = Tr.GetElement1IntPoint();

            // Compute unit normal and surface measure
            if (!Tr.Face) { continue; }
            Tr.Face->SetIntPoint(&ip);
            const mfem::DenseMatrix &J = Tr.Face->Jacobian();
            mfem::CalcOrtho(J, normal);
            double dS = normal.Norml2();
            if (dS > 0.0) { normal /= dS; }

            // Physical coordinate (optional for inflow value)
            xphys.SetSize(dim_);
            Tr.Face->Transform(ip, xphys);

            // Inflow factor: max( - v·n, 0 )
            const double wdotn = vg_ * (omega_[0]*normal[0]
                + (dim_ > 1 ? omega_[1]*normal[1] : 0.0)
                + (dim_ > 2 ? omega_[2]*normal[2] : 0.0));
            if (wdotn >= 0.0) { continue; }

            double g_in = 0.0;
            if (bdr_map_ && Tr.Face)
            {
                const int attr = Tr.Face->Attribute;
                auto it = bdr_map_->find(attr);
                if (it != bdr_map_->end())
                {
                    g_in = it->second.wallTemperature; // already mapped to BE(Tw) by caller if needed
                }
            }
            if ((!bdr_map_) && inflow_value_)
            {
                g_in = inflow_value_(omega_, xphys);
            }
            const double w = (-wdotn) * dS * ip.weight;

            // Test shapes on element side-1
            el.CalcShape(eip, shape);
            for (int j = 0; j < shape.Size(); j++)
            {
                elvect[j] += w * g_in * shape[j];
            }
        }
    }

    // Satisfy pure virtual for non-face variant (not used here).
    void AssembleRHSElementVect(const mfem::FiniteElement &el,
                                mfem::ElementTransformation &Tr,
                                mfem::Vector &elvect) override
    {
        elvect.SetSize(el.GetDof());
        elvect = 0.0;
        (void)Tr;
    }

private:
    int dim_;
    double vg_;
    std::array<double,3> omega_;
    const BoundaryConditionMap* bdr_map_;
    InflowValue inflow_value_;
};

} // namespace dg


