#pragma once

namespace DGSolver {

    template<int Dim>
    class DGSolver {
        public:
            DGSolver() = default;
            virtual ~DGSolver() = default;
    };

} // namespace DGSolver