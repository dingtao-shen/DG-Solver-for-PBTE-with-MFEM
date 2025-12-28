#pragma once

namespace PhononModel {

    template<int SpatialDim>
    class PhononModel {
        public:
            PhononModel() = default;
            virtual ~PhononModel() = default;
    };

}