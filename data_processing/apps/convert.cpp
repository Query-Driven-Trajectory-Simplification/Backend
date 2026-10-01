#include <cmath>
#include <iostream>
#include <string>

#include "backend/data/readers/geolife.hpp"

using geolife::parse_plt_line;


int main() {
    const std::string good = "39.984702,116.318417,0,492,39744.1201851852,2008-10-23,02:53:04";

    auto p = parse_plt_line(good);
    std::cout << p->lat << ' ' << p->lon << ' ' << p->t;
}
