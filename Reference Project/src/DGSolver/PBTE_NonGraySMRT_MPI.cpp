#include "DGSolver/PBTE_NonGraySMRT.hpp"
#include <limits>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace DGSolver {

    template<int dim>
    PBTE_NonGraySMRT_MPI<dim>::PBTE_NonGraySMRT_MPI(std::vector<std::shared_ptr<SpatialMesh::Cell<dim, dim+1>>>& Cells,
                                const SpatialMesh::MeshPartitionInfo<dim>& partition_info,
                                const PhononModel::NonGraySMRT_MPI<dim>& non_gray_smrt_mpi,
                                MPI_Comm communicator) : comm(communicator) 
    {
        omp_set_num_threads(omp_get_max_threads());                        
        MPI_Comm_rank(comm, &world_rank);
        MPI_Comm_size(comm, &world_size);

        const auto& local_cell_ids = partition_info.getPartitionCells(world_rank);
        for(const auto& CellID : local_cell_ids){
            local_source.insert({CellID, std::vector<std::vector<Eigen::VectorXd>>(
                CC.POLAR_DIM, std::vector<Eigen::VectorXd>(
                CC.NSPEC, Eigen::VectorXd::Zero(CC.DOF)))});
        }
        int n_local_cells = partition_info.getPartitionCells(world_rank).size();
        int n_nbr_cells = partition_info.getPartitionNbrCells(world_rank).size();
        local_solutions.resize(n_local_cells * CC.POLAR_DIM * CC.NSPEC * CC.NPOLE * CC.NAZIM * CC.DOF);
        nbr_solutions.resize(n_nbr_cells * CC.POLAR_DIM * CC.NSPEC * CC.NPOLE * CC.NAZIM * CC.DOF);
        local_Tc.resize(n_local_cells * CC.DOF);

        dt_inv = 0.0;
        for(int p = 0; p < CC.POLAR_DIM; p++){
            dt_inv = std::max(non_gray_smrt_mpi.getInvKn(p).maxCoeff(), dt_inv);
        }

        residual = 1000000.0;
        iter_step = 0;
        const long long msg_size_ll = 1LL * CC.POLAR_DIM * CC.NSPEC * CC.NPOLE * CC.NAZIM * CC.DOF;
        if (msg_size_ll > std::numeric_limits<int>::max()) {
            throw std::runtime_error("MPI message size exceeds int range; reduce degrees or change packing");
        }
        single_cell_msg_size = static_cast<int>(msg_size_ll);

        // Removed barrier to allow overlap of communication and computation
        precomputed_lu_ready = false;
    }

    template<int dim>
    PBTE_NonGraySMRT_MPI<dim>::~PBTE_NonGraySMRT_MPI() {
        local_source.clear();
        local_solutions.clear();
        nbr_solutions.clear();
        local_Tc.clear();
    }

    template<int dim>
    void PBTE_NonGraySMRT_MPI<dim>::exchange_solutions(const SpatialMesh::MeshPartitionInfo<dim>& partition_info) {
        // Bulk exchange: one message per neighbor rank containing all local cell solutions needed by that rank
        // Use non-blocking communication to avoid deadlock
        const auto& local_cell_idx = partition_info.getLocalCellIdx(world_rank);
        // Use precomputed maps from partition_info
        const auto& dest_to_cells = partition_info.getPartitionDestCells(world_rank);
        const auto& source_ranks = partition_info.getPartitionSourceRanks(world_rank);
        const auto& recv_cells = partition_info.getPartitionRecvCells(world_rank);
        const auto& recv_indices = partition_info.getPartitionRecvNbrIndices(world_rank);

        // Prepare send buffers and post non-blocking sends first (reuse persistent per-rank buffers)
        std::vector<MPI_Request> send_requests;
        send_requests.reserve(dest_to_cells.size());

        for (const auto& kv : dest_to_cells) {
            const int dest_rank = kv.first;
            const auto& cell_ids = kv.second;
            const int count = cell_ids.size();
            if (count == 0) continue;
            auto& sbuf = send_buffers[dest_rank];
            const size_t buf_elems = static_cast<size_t>(count) * static_cast<size_t>(single_cell_msg_size);
            // grow-only resize to avoid repeated zero-fill; actual send count is set explicitly
            if (sbuf.size() < buf_elems) {
                sbuf.resize(buf_elems);
            }
            // Precompute local indices to avoid hash lookups in parallel region
            std::vector<int> local_indices;
            local_indices.reserve(static_cast<size_t>(count));
            for (int i = 0; i < count; ++i) {
                local_indices.push_back(local_cell_idx.at(cell_ids[i]));
            }
            // Pack in parallel
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < count; ++i) {
                const int lidx = local_indices[i];
                const size_t src_offset = static_cast<size_t>(lidx) * static_cast<size_t>(single_cell_msg_size);
                std::memcpy(sbuf.data() + static_cast<size_t>(i) * static_cast<size_t>(single_cell_msg_size),
                            local_solutions.data() + src_offset,
                            sizeof(double) * static_cast<size_t>(single_cell_msg_size));
            }
            MPI_Request req;
            MPI_Isend(sbuf.data(),
                      count * single_cell_msg_size,
                      MPI_DOUBLE,
                      dest_rank,
                      90000,
                      comm,
                      &req);
            send_requests.push_back(req);
        }

        // Post non-blocking receives from each source rank (reuse persistent per-rank buffers)
        std::vector<MPI_Request> recv_requests;
        std::vector<int> recv_sources;
        recv_requests.reserve(source_ranks.size());
        recv_sources.reserve(source_ranks.size());

        for (const int source_rank : source_ranks) {
            auto it_cells = recv_cells.find(source_rank);
            if (it_cells == recv_cells.end()) continue;
            const auto& remote_ids = it_cells->second;
            const int count = remote_ids.size();
            if (count <= 0) continue;
            auto& rbuf = recv_buffers[source_rank];
            const size_t buf_elems = static_cast<size_t>(count) * static_cast<size_t>(single_cell_msg_size);
            // grow-only resize to avoid repeated zero-fill; actual recv count is explicit
            if (rbuf.size() < buf_elems) {
                rbuf.resize(buf_elems);
            }
            MPI_Request req;
            MPI_Irecv(rbuf.data(),
                      count * single_cell_msg_size,
                      MPI_DOUBLE,
                      source_rank,
                      90000,
                      comm,
                      &req);
            recv_sources.push_back(source_rank);
            recv_requests.push_back(req);
        }

        // Process receives as they complete and scatter into nbr_solutions
        if (!recv_requests.empty()) {
            const int nreq = static_cast<int>(recv_requests.size());
            int completed = 0;
            std::vector<int> indices(static_cast<size_t>(nreq));
            while (completed < nreq) {
                int outcount = 0;
                MPI_Waitsome(nreq, recv_requests.data(), &outcount, indices.data(), MPI_STATUSES_IGNORE);
                if (outcount == MPI_UNDEFINED || outcount == 0) {
                    continue;
                }
                for (int k = 0; k < outcount; ++k) {
                    const int idx_req = indices[static_cast<size_t>(k)];
                    if (idx_req == MPI_UNDEFINED) continue;
                    const int source_rank = recv_sources[static_cast<size_t>(idx_req)];
                    auto it_indices = recv_indices.find(source_rank);
                    if (it_indices == recv_indices.end()) continue;
                    const auto& nbr_indices = it_indices->second;
                    const auto it_cells = recv_cells.find(source_rank);
                    if (it_cells == recv_cells.end()) continue;
                    const auto& data_buf = recv_buffers[source_rank];
                    const size_t expected_count = nbr_indices.size();
                    if (expected_count * static_cast<size_t>(single_cell_msg_size) > data_buf.size()) {
                        throw std::runtime_error("Received buffer size less than expected neighbor count");
                    }
                    #pragma omp parallel for schedule(static)
                    for (long long j = 0; j < static_cast<long long>(expected_count); ++j) {
                        const int nbr_idx = nbr_indices[static_cast<size_t>(j)];
                        const size_t dst_offset = static_cast<size_t>(nbr_idx) * static_cast<size_t>(single_cell_msg_size);
                        std::memcpy(nbr_solutions.data() + dst_offset,
                                    data_buf.data() + static_cast<size_t>(j) * static_cast<size_t>(single_cell_msg_size),
                                    sizeof(double) * static_cast<size_t>(single_cell_msg_size));
                    }
                }
                completed += outcount;
            }
        }

        // Wait for all sends to complete
        if (!send_requests.empty()) {
            MPI_Waitall(static_cast<int>(send_requests.size()), send_requests.data(), MPI_STATUSES_IGNORE);
        }
    }

    template<int dim>
    void PBTE_NonGraySMRT_MPI<dim>::gather_solutions(const SpatialMesh::MeshPartitionInfo<dim>& partition_info, PhononModel::NonGraySMRT_MPI<dim>& non_gray_smrt) {
        // Ensure all ranks have finished local computations
        // Removed barrier to avoid global synchronization during construction

        const size_t cell_data_size = static_cast<size_t>(single_cell_msg_size);

        if (world_rank == 0) {
            const auto& local_cell_ids = partition_info.getPartitionCells(world_rank);
            // Update root's own cells first
            for (int idx = 0; idx < local_cell_ids.size(); ++idx) {
                const int CellID = local_cell_ids[idx];
                const size_t cell_offset = idx * cell_data_size;
                
                // Parallelize the nested loops over (p, s, j1, j2)
                #pragma omp parallel for collapse(4) schedule(static)
                for (int p = 0; p < CC.POLAR_DIM; p++) {
                    for (int s = 0; s < CC.NSPEC; s++) {
                        for (int j1 = 0; j1 < CC.NPOLE; j1++) {
                            for (int j2 = 0; j2 < CC.NAZIM; j2++) {
                                // Calculate offset for this (p, s, j1, j2) in the serialized array
                                const size_t offset = cell_offset + 
                                    p * CC.NSPEC * CC.NPOLE * CC.NAZIM * CC.DOF +
                                    s * CC.NPOLE * CC.NAZIM * CC.DOF +
                                    j1 * CC.NAZIM * CC.DOF +
                                    j2 * CC.DOF;
                                // Use Eigen::Map to create a view of the data without copying
                                Eigen::Map<const Eigen::VectorXd> solution_vec(
                                    local_solutions.data() + offset, CC.DOF);
                                non_gray_smrt.updateApproxCoeff(CellID, p, s, j1, j2, solution_vec);
                            }
                        }
                    }
                }
            }

            // Receive one contiguous local_solutions buffer per source rank and scatter
            for (int source_rank = 1; source_rank < world_size; ++source_rank) {
                if (source_rank >= partition_info.getNumPartitions()) {
                    break; // no more partition-owning ranks
                }
                const auto& source_cells = partition_info.getPartitionCells(source_rank);
                if (source_cells.empty()) continue;
                const int recv_count = static_cast<int>(source_cells.size()) * single_cell_msg_size;
                std::vector<double> recv_buffer(static_cast<size_t>(recv_count));
                const int gather_tag = 50000 + source_rank; // distinct tag space
                MPI_Recv(recv_buffer.data(), recv_count, MPI_DOUBLE, source_rank, gather_tag, comm, MPI_STATUS_IGNORE);

                // Scatter: for each cell index in source partition, map its block and update
                #pragma omp parallel for schedule(static)
                for (int idx = 0; idx < static_cast<int>(source_cells.size()); ++idx) {
                    const int CellID = source_cells[idx];
                    const size_t cell_base = static_cast<size_t>(idx) * static_cast<size_t>(single_cell_msg_size);
                    for (int p = 0; p < CC.POLAR_DIM; p++) {
                        for (int s = 0; s < CC.NSPEC; s++) {
                            for (int j1 = 0; j1 < CC.NPOLE; j1++) {
                                for (int j2 = 0; j2 < CC.NAZIM; j2++) {
                                    const size_t offset = cell_base +
                                        static_cast<size_t>(p) * CC.NSPEC * CC.NPOLE * CC.NAZIM * CC.DOF +
                                        static_cast<size_t>(s) * CC.NPOLE * CC.NAZIM * CC.DOF +
                                        static_cast<size_t>(j1) * CC.NAZIM * CC.DOF +
                                        static_cast<size_t>(j2) * CC.DOF;
                                    Eigen::Map<const Eigen::VectorXd> solution_vec(
                                        recv_buffer.data() + offset, CC.DOF);
                                    non_gray_smrt.updateApproxCoeff(CellID, p, s, j1, j2, solution_vec);
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // Only ranks that own actual partitions send
            if (world_rank < partition_info.getNumPartitions()) {
                // Send entire local_solutions buffer in one message; ordered by local index
                const auto& local_cells = partition_info.getPartitionCells(world_rank);
                if (!local_cells.empty()) {
                    const int send_count = static_cast<int>(local_cells.size()) * single_cell_msg_size;
                    const int gather_tag = 50000 + world_rank;
                    MPI_Send(const_cast<double*>(local_solutions.data()), send_count, MPI_DOUBLE, 0, gather_tag, comm);
                }
            }
        }
    }

    template<int dim>
    void PBTE_NonGraySMRT_MPI<dim>::update_Tc(const SpatialMesh::MeshPartitionInfo<dim>& partition_info, PhononModel::NonGraySMRT_MPI<dim>& non_gray_smrt, Eigen::MatrixXd& IntMat) {

        // Root receives from other ranks and fills NonGraySMRT::Tc, non-root sends
        if (world_rank == 0) {
            // First set root owned cell columns
            const auto& local_cell_ids = partition_info.getPartitionCells(world_rank);
            for (int idx = 0; idx < local_cell_ids.size(); ++idx) {
                const Eigen::VectorXd& col = Eigen::Map<const Eigen::VectorXd>(local_Tc.data() + idx * CC.DOF, CC.DOF);
                non_gray_smrt.update_global_Tc(local_cell_ids[idx], col);
            }

            // Receive one contiguous local_Tc buffer per source rank and scatter by offset
            for (int source_rank = 1; source_rank < world_size; ++source_rank) {
                if (source_rank >= partition_info.getNumPartitions()) break;
                const auto& source_cells = partition_info.getPartitionCells(source_rank);
                if (source_cells.empty()) continue;
                const int recv_count = static_cast<int>(source_cells.size() * CC.DOF);
                std::vector<double> recv_buf(static_cast<size_t>(recv_count));
                // Use a fixed tag space for update_Tc distinct from gather_solutions
                const int tag = 60000 + source_rank;
                MPI_Recv(recv_buf.data(), recv_count, MPI_DOUBLE, source_rank, tag, comm, MPI_STATUS_IGNORE);
                // Scatter by offset: index in source_cells corresponds to local_cell_idx on sender
                for (size_t idx = 0; idx < source_cells.size(); ++idx) {
                    const int CellID = source_cells[idx];
                    const double* ptr = recv_buf.data() + idx * CC.DOF;
                    Eigen::Map<const Eigen::VectorXd> vec(ptr, CC.DOF);
                    non_gray_smrt.update_global_Tc(CellID, vec);
                }
            }

            residual = non_gray_smrt.cal_residual_MPI(IntMat);
            MPI_Bcast(&residual, 1, MPI_DOUBLE, 0, comm);
        } else {
            if (world_rank < partition_info.getNumPartitions()) {
                // Send entire local_Tc buffer in one message; it is ordered by local_cell_idx
                const auto& local_cells = partition_info.getPartitionCells(world_rank);
                if (!local_cells.empty()) {
                    const int send_count = static_cast<int>(local_cells.size() * CC.DOF);
                    // local_Tc is already sized as n_local_cells * DOF and laid out by local index
                    const int tag = 60000 + world_rank;
                    MPI_Send(const_cast<double*>(local_Tc.data()), send_count, MPI_DOUBLE, 0, tag, comm);
                }
            }
            residual = 0.0;
            MPI_Bcast(&residual, 1, MPI_DOUBLE, 0, comm);
        }
    }

    template<int dim>
    void PBTE_NonGraySMRT_MPI<dim>::solve(std::vector<std::shared_ptr<SpatialMesh::Cell<dim, dim+1>>>& Cells,
                                        const SpatialMesh::MeshPartitionInfo<dim>& partition_info,
                                        Eigen::MatrixXd& IntMat,
                                        std::vector<Eigen::MatrixXd>& MassMat, 
                                        std::vector<std::vector<Eigen::MatrixXd>>& MassOnFace, 
                                        std::vector<std::vector<Eigen::MatrixXd>>& StfMat,
                                        std::vector<std::vector<Eigen::MatrixXd>>& FluxInt,
                                        PhononModel::NonGraySMRT_MPI<dim>& non_gray_smrt,
                                        SolidAngle& dsa, 
                                        std::vector<std::vector<Eigen::MatrixXd>>& outflow)
    {
        omp_set_num_threads(omp_get_max_threads());
        const auto& local_cell_ids = partition_info.getPartitionCells(world_rank);
        const auto& local_cell_idx = partition_info.getLocalCellIdx(world_rank);

        // Precompute LU once for all local cells and angle/spec combinations
        if (!precomputed_lu_ready) {
            const int n_local_cells = static_cast<int>(local_cell_ids.size());
            PreComputedLU_local.resize(CC.POLAR_DIM, std::vector<std::vector<std::vector<std::vector<Eigen::PartialPivLU<Eigen::MatrixXd>>>>>(
                CC.NSPEC, std::vector<std::vector<std::vector<Eigen::PartialPivLU<Eigen::MatrixXd>>>>(
                CC.NPOLE, std::vector<std::vector<Eigen::PartialPivLU<Eigen::MatrixXd>>>(
                CC.NAZIM, std::vector<Eigen::PartialPivLU<Eigen::MatrixXd>>(
                n_local_cells, Eigen::PartialPivLU<Eigen::MatrixXd>(Eigen::MatrixXd::Zero(CC.DOF, CC.DOF)))))));
            
            const auto& dir_all = dsa.dir();
            #pragma omp parallel for collapse(4) schedule(static)
            for (int p = 0; p < CC.POLAR_DIM; ++p) {
                for (int s = 0; s < CC.NSPEC; ++s) {
                    for (int j1 = 0; j1 < CC.NPOLE; ++j1) {
                        for (int j2 = 0; j2 < CC.NAZIM; ++j2) {
                            const double vg = non_gray_smrt.getVg(p, s);
                            const auto dir = dir_all[j1][j2];
                            for (int loc = 0; loc < n_local_cells; ++loc) {
                                const int CellID = local_cell_ids[loc];
                                Eigen::MatrixXd Coeff = dt_inv * MassMat[CellID];
                                for (int i = 0; i < dim; ++i) {
                                    Coeff.noalias() -= vg * dir(i) * StfMat[CellID][i];
                                }
                                for (int il = 0; il < dim + 1; ++il) {
                                    const double f = outflow[j1][j2](il, CellID);
                                    const double fp = 0.5 * vg * (f + std::abs(f));
                                    if (fp != 0.0) {
                                        Coeff.noalias() += fp * MassOnFace[CellID][il];
                                    }
                                }
                                PreComputedLU_local[p][s][j1][j2][loc] = Coeff.lu();
                            }
                        }
                    }
                }
            }
            precomputed_lu_ready = true;
        }

        // Build lookup sets and static computation order once
        const auto& local_nbr_cell_idx = partition_info.getLocalNbrCellIdx(world_rank);
        const auto& nbr_cells = partition_info.getPartitionNbrCells(world_rank);
        std::unordered_set<int> local_cell_set(local_cell_ids.begin(), local_cell_ids.end());
        std::unordered_set<int> nbr_cell_set(nbr_cells.begin(), nbr_cells.end());
        const auto& partition_comp_order = partition_info.getPartitionComputationOrder(world_rank);

        while(residual > CC.TOL && iter_step < CC.TMAX){
            // Recompute source term from current local_Tc at the start of each iteration
            #pragma omp parallel for schedule(static)
            for(size_t idx = 0; idx < local_cell_ids.size(); ++idx){
                const int CellID = local_cell_ids[idx];
                const int local_tc_offset = idx * CC.DOF;
                Eigen::Map<const Eigen::VectorXd> tc_vec(local_Tc.data() + local_tc_offset, CC.DOF);
                for(int p = 0; p < CC.POLAR_DIM; p++){
                    for(int s = 0; s < CC.NSPEC; s++){
                        const double Kn_inv = non_gray_smrt.getInvKn(p, s);
                        const double Cwp = non_gray_smrt.getHeatCap(p, s);
                        local_source[CellID][p][s] = Kn_inv * Cwp / CC.OMEGA * (tc_vec.transpose() * MassMat[CellID]).transpose();
                    }
                }
            }
            // Reset local_Tc accumulator for this iteration
            std::fill(local_Tc.begin(), local_Tc.end(), 0.0);

            exchange_solutions(partition_info);

            const size_t local_Tc_size = local_Tc.size();
            const int num_threads = omp_get_max_threads();
            std::vector<std::vector<double>> thread_Tc_acc(static_cast<size_t>(num_threads), std::vector<double>(local_Tc_size, 0.0));

            #pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                double* tc_acc = thread_Tc_acc[static_cast<size_t>(tid)].data();
                Eigen::VectorXd rhs(CC.DOF);
                Eigen::VectorXd sol(CC.DOF);
                #pragma omp for collapse(4) schedule(static)
                for(int j1 = 0; j1 < CC.NPOLE; j1++){
                for(int j2 = 0; j2 < CC.NAZIM; j2++){
                for(int p = 0; p < CC.POLAR_DIM; p++){
                for(int s = 0; s < CC.NSPEC; s++){
                    // Get computation order for this angle combination
                    const auto& comp_order = partition_comp_order[j1][j2];
                    
                    // Process cells in computation order
                    for(size_t k = 0; k < comp_order.size(); ++k){
                        const int CellID = comp_order[k];
                        // Find local index for this cell
                        auto it = local_cell_idx.find(CellID);
                        if(it == local_cell_idx.end()) continue; // Skip if cell not in this partition
                        const size_t idx = static_cast<size_t>(it->second);
                        
                        const double Cwp = non_gray_smrt.getHeatCap(p, s);
                        const double vg = non_gray_smrt.getVg(p, s);
                        const double factor_dt = dt_inv - non_gray_smrt.getInvKn(p, s);
                        const auto dir = dsa.dir()[j1][j2];

                        // Access approx_coeff from serialized local_solutions
                        const size_t approx_offset = idx * static_cast<size_t>(single_cell_msg_size) + 
                            static_cast<size_t>(p) * CC.NSPEC * CC.NPOLE * CC.NAZIM * CC.DOF +
                            static_cast<size_t>(s) * CC.NPOLE * CC.NAZIM * CC.DOF +
                            static_cast<size_t>(j1) * CC.NAZIM * CC.DOF +
                            static_cast<size_t>(j2) * CC.DOF;
                        Eigen::Map<const Eigen::VectorXd> approx_coeff(local_solutions.data() + approx_offset, CC.DOF);

                        rhs = local_source[CellID][p][s]; 
                        rhs.noalias() += factor_dt * (MassMat[CellID].transpose() * approx_coeff);

                        // boundary condition
                        for(int il = 0; il < dim+1; il++){
                            int BCTAG = Cells[CellID]->getFaces()[il]->getBoundaryTag();
                            double f = outflow[j1][j2](il, CellID);
                            const double f_abs = std::abs(f);
                            if(BCTAG == 0){
                                int NBR = (Cells[CellID]->getFaces()[il]->getAdjacentCells()[0] == CellID) ? Cells[CellID]->getFaces()[il]->getAdjacentCells()[1] : Cells[CellID]->getFaces()[il]->getAdjacentCells()[0];
                                if(local_cell_set.find(NBR) != local_cell_set.end()){
                                    // NBR is a local cell
                                    const int nbr_idx = local_cell_idx.at(NBR);
                                    const size_t nbr_offset = static_cast<size_t>(nbr_idx) * static_cast<size_t>(single_cell_msg_size) + 
                                        static_cast<size_t>(p) * CC.NSPEC * CC.NPOLE * CC.NAZIM * CC.DOF +
                                        static_cast<size_t>(s) * CC.NPOLE * CC.NAZIM * CC.DOF +
                                        static_cast<size_t>(j1) * CC.NAZIM * CC.DOF +
                                        static_cast<size_t>(j2) * CC.DOF;
                                    Eigen::Map<const Eigen::VectorXd> nbr_sol(local_solutions.data() + nbr_offset, CC.DOF);
                                    const double coeff_in = 0.5 * vg * (f - f_abs);
                                    if (coeff_in != 0.0) {
                                        rhs.noalias() -= coeff_in * (FluxInt[CellID][il] * nbr_sol);
                                    }
                                }
                                else if(nbr_cell_set.find(NBR) != nbr_cell_set.end()){
                                    // NBR is a neighbor cell
                                    const int nbr_idx = local_nbr_cell_idx.at(NBR);
                                    const size_t nbr_offset = static_cast<size_t>(nbr_idx) * static_cast<size_t>(single_cell_msg_size) + 
                                        static_cast<size_t>(p) * CC.NSPEC * CC.NPOLE * CC.NAZIM * CC.DOF +
                                        static_cast<size_t>(s) * CC.NPOLE * CC.NAZIM * CC.DOF +
                                        static_cast<size_t>(j1) * CC.NAZIM * CC.DOF +
                                        static_cast<size_t>(j2) * CC.DOF;
                                    Eigen::Map<const Eigen::VectorXd> nbr_sol(nbr_solutions.data() + nbr_offset, CC.DOF);
                                    const double coeff_in = 0.5 * vg * (f - f_abs);
                                    if (coeff_in != 0.0) {
                                        rhs.noalias() -= coeff_in * (FluxInt[CellID][il] * nbr_sol);
                                    }
                                }
                                else {
                                    std::cout << ">>> Process " << world_rank << ": NBR " << NBR << " not found in local_solutions or nbr_solutions" << std::endl;
                                    std::cout << partition_info.getCellPartition(NBR) << std::endl;
                                    throw std::runtime_error("NBR not found in local_solutions or nbr_solutions");
                                }
                            }
                            else if(CC.BOUNDARY_COND[BCTAG].first == 1){ // Thermalizing boundary condition
                                const double coeff_in = 0.5 * vg * (f - f_abs);
                                if (coeff_in != 0.0) {
                                    rhs.noalias() -= coeff_in * Cwp / CC.OMEGA * CC.BOUNDARY_COND[BCTAG].second * FluxInt[CellID][il].col(0);
                                }
                            }
                            else {
                                throw std::runtime_error("Invalid boundary condition");
                            }
                        }
                        const int loc = static_cast<int>(idx);
                        const auto& lu = PreComputedLU_local[p][s][j1][j2][loc];
                        sol = lu.solve(rhs);
                        const size_t local_solution_offset = idx * static_cast<size_t>(single_cell_msg_size) + static_cast<size_t>(p) * CC.NSPEC * CC.NPOLE * CC.NAZIM * CC.DOF + static_cast<size_t>(s) * CC.NPOLE * CC.NAZIM * CC.DOF + static_cast<size_t>(j1) * CC.NAZIM * CC.DOF + static_cast<size_t>(j2) * CC.DOF;
                        std::memcpy(local_solutions.data() + local_solution_offset, sol.data(), sizeof(double) * CC.DOF);
                        
                        // Accumulate Tc contribution for this (cell, p, s, j1, j2) into thread-local buffer
                        const size_t tc_offset = idx * static_cast<size_t>(CC.DOF);
                        const double scale = (1.0 / non_gray_smrt.getHeatCapV()) * non_gray_smrt.getInvKn(p, s) * dsa.wt()[j1][j2] * non_gray_smrt.getW(p)(1, s);
                        for (int d = 0; d < CC.DOF; ++d) {
                            tc_acc[tc_offset + static_cast<size_t>(d)] += scale * sol[d];
                        }
                    } // end for k (cells in computation order)
                }}}} // end collapse(4)
            } // end parallel region
            // Merge thread-local accumulators into local_Tc
            for (int t = 0; t < num_threads; ++t) {
                const double* src = thread_Tc_acc[static_cast<size_t>(t)].data();
                for (size_t i = 0; i < local_Tc_size; ++i) {
                    local_Tc[i] += src[i];
                }
            }

            // std::cout << ">>> Process " << world_rank << ": local_Tc calculation completed" << std::endl;

            update_Tc(partition_info, non_gray_smrt, IntMat);
            iter_step++;
            if(world_rank == 0){
                std::cout << ">>> Iteration " << iter_step << ", residual = " << residual << std::endl;
                std::cout << "Tv.norm() = " << non_gray_smrt.getGlobalTv().norm() << std::endl;
            }
        }

        gather_solutions(partition_info, non_gray_smrt);
    }

}

template class DGSolver::PBTE_NonGraySMRT_MPI<2>;
template class DGSolver::PBTE_NonGraySMRT_MPI<3>;