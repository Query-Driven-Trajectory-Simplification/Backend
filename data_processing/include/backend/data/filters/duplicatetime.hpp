#pragma once

#include "backend/data/readers/geolife.hpp"

namespace filters {

geolife::Segment drop_duplicate_times(geolife::Segment segment);

}
