# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2015-2019 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

"""
Module exports :class:`VacareanuEtAl2015`
"""
import numpy as np

from openquake.hazardlib.gsim.base import GMPE, CoeffsTable
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGA, SA


class VacareanuEtAl2015(GMPE):
    """
    Implements the Vrancea intermediate depth GMPE developed by R. Vacareanu, M. Radulian,
    M. Iancovici, F. Pavel and C. Neagu, published as "Fore-Arc and Back-Arc Ground Motion
    Model for Vrancea Intermediate Depth Seismic Source" (2015, Journal of Earthquake Engineering).
    To be completed
    """

    #: Supported tectonic region type is subduction interface
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.VRANCEA_INTERMEDIATE

    #: Supported intensity measure types are spectral acceleration,
    #: and peak ground acceleration
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set([
        PGA,
        SA
    ])

    #: Supported intensity measure component is the geometric mean component
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.AVERAGE_HORIZONTAL

    #: Supported standard deviation types are inter-event, intra-event
    #: and total, see table 3, pages 12 - 13
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = set([
        const.StdDev.TOTAL,
        const.StdDev.INTER_EVENT,
        const.StdDev.INTRA_EVENT
    ])

    #: Site amplification is dependent upon the site class per EN 1998-1
    #: and thus to the Vs30 site velocity parameter
    #: For the Vacareanu et al (2015) GMPE a new term is introduced to
    #: determine whether a site is on the forearc with respect to the
    #: subduction interface, or on the backarc. This boolean is a vector
    #: containing True for a backarc site or False for a forearc or
    #: unknown site.

    REQUIRES_SITES_PARAMETERS = set(('vs30', 'backarc'))

    #: Required rupture parameters are magnitude for the interface model
    REQUIRES_RUPTURE_PARAMETERS = set(('mag', 'hypo_depth'))

    #: Required distance measure is the hypocentral distance, for
    #: Vrancea Intermediate Depth events
    REQUIRES_DISTANCES = set(('rhypo',))

    #: Lower boundary Vs30 values representing EN 1998-1 classes:
    #: SoilA_VS30 = 800
    #: SoilB_VS30 = 360
    #: SoilC_VS30 = 180

    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
        for spec of input and result values.
        """
        # extract dictionaries of coefficients specific to required
        # intensity measure type and for PGA

        C = self.COEFFS[imt]

        mean = (self._compute_magnitude_term(C, rup.mag) +
                self._compute_distance_arc_term(C, rup.mag, sites, dists) +
                self._compute_focal_depth_term(C, rup) +
                self._compute_site_response_term(C, sites))
        stddevs = self._get_stddevs(C, stddev_types)
        return mean, stddevs

    def _compute_magnitude_term(self, C, mag):
        """"
        Computes the magnitude scaling term given by equation (1)
        """
        return C['c1'] + C['c2'] * (mag - 6) + C['c3'] * (mag - 6) * (mag - 6)

    def _compute_distance_arc_term(self, C, mag, sites, dists, ):
        """
        Computes the distance and fore-arc/back-arc scaling term, as contained within equation (1)
        """
        return (C['c4'] * np.log(dists.rhypo) +
                C['c5'] * sites.backarc * dists.rhypo +
                C['c6'] * (1 - sites.backarc) * dists.rhypo)

    def _compute_focal_depth_term(self, C, rup):
        """
        Computes the hypocentral depth scaling term - as indicated by
        equation (1)
        """
        return C['c7'] * rup.hypo_depth

    def _compute_site_response_term(self, C, sites):
        """
        Compute and return site response model term
        This GMPE uses specific site model scaling factors
        """
        site_b = 0
        site_c = 0
        site_s = 0

        vs_star = sites.vs30.copy()
        vs_star[vs_star > 800.0] = 800.
        vs_star[vs_star < 180.0] = 180.

        # Set site class from Vs30
        if vs_star <= 360.0:
            site_c = 1.0
        else:
            site_b = 1.0

        return C['c8'] * site_b + C['c9'] * site_c + C['c10'] * site_s

    def _get_stddevs(self, C, stddev_types):
        """
        Return standard deviations as defined in Table 4
        """
        stddevs = []
        for stddev_type in stddev_types:
            assert stddev_type in self.DEFINED_FOR_STANDARD_DEVIATION_TYPES
            if stddev_type == const.StdDev.TOTAL:
                stddevs.append(C['sigma_t'])
            elif stddev_type == const.StdDev.INTER_EVENT:
                stddevs.append(C['tau'])
            elif stddev_type == const.StdDev.INTRA_EVENT:
                stddevs.append(C['sigma'])

        return stddevs

    # Period-dependent coefficients (Table 4)
    COEFFS = CoeffsTable(sa_damping=5, table="""\
    imt      c1     c2      c3      c4      c5      c6      c7      c8     c9     c10 sigma_t   tau sigma  
    0.0  9.6231 1.4232 −0.1555 −1.1316 −0.0114 −0.0024 −0.0007 −0.0835 0.1589  0.0488   0.698 0.406 0.568
    0.1  9.6981 1.3679 −0.1423 −0.9889 −0.0135 −0.0026 −0.0017 −0.1965 0.1670  0.0020   0.806 0.468 0.656
    0.2 10.0090 1.3620 −0.1138 −1.0371 −0.0127 −0.0032 −0.0004 −0.1547 0.2861  0.0860   0.792 0.469 0.638
    0.3 10.7033 1.4580 −0.1187 −1.2340 −0.0106 −0.0026  0.0000 −0.1014 0.2659  0.0991   0.783 0.480 0.619
    0.4 10.7701 1.5748 −0.1439 −1.3207 −0.0093 −0.0022  0.0005 −0.1076 0.3062  0.1183   0.810 0.519 0.622
    0.5  9.2327 1.6739 −0.1664 −1.0022 −0.0100 −0.0041  0.0007 −0.0259 0.2576  0.0722   0.767 0.461 0.613
    0.6  8.6445 1.7672 −0.1925 −0.8938 −0.0099 −0.0045 −0.0004 −0.1038 0.2181  0.0179   0.740 0.429 0.603
    0.7  8.7134 1.8500 −0.1990 −0.9780 −0.0088 −0.0039  0.0002 −0.1867 0.1564  0.0006   0.735 0.426 0.599
    0.8  9.0835 1.9066 −0.2022 −1.1044 −0.0078 −0.0031  0.0005 −0.2901 0.0546 −0.1019   0.726 0.417 0.594
    0.9  9.1274 1.9662 −0.2465 −1.1437 −0.0074 −0.0031  0.0001 −0.2804 0.0884 −0.0790   0.719 0.403 0.596
    1.0  8.9987 1.9964 −0.2658 −1.1226 −0.0071 −0.0031 −0.0009 −0.2992 0.0739 −0.0955   0.715 0.400 0.592
    1.2  8.0465 2.0432 −0.2241 −0.9654 −0.0072 −0.0041 −0.0013 −0.2681 0.1476 −0.0412   0.713 0.392 0.595
    1.4  7.0585 2.1148 −0.2167 −0.8011 −0.0078 −0.0049 −0.0013 −0.2566 0.2009 −0.0068   0.714 0.392 0.597
    1.6  6.8329 2.1668 −0.2418 −0.8036 −0.0075 −0.0047 −0.0018 −0.2268 0.2272  0.0211   0.732 0.418 0.601
    1.8  6.4292 2.1988 −0.2468 −0.7625 −0.0073 −0.0047 −0.0020 −0.2464 0.2200  0.0082   0.745 0.427 0.611
    2.0  6.3876 2.2151 −0.2289 −0.8004 −0.0066 −0.0043 −0.0024 −0.2767 0.2134 −0.0091   0.744 0.425 0.611
    2.5  4.4248 2.2541 −0.2144 −0.4280 −0.0079 −0.0061 −0.0031 −0.2924 0.2108 −0.0177   0.750 0.420 0.622
    3.0  4.5395 2.2812 −0.2256 −0.5340 −0.0072 −0.0054 −0.0034 −0.3066 0.1840 −0.0387   0.765 0.436 0.629
    3.5  4.7407 2.2803 −0.2456 −0.6250 −0.0065 −0.0045 −0.0041 −0.3728 0.0918 −0.1192   0.778 0.436 0.645
    4.0  4.4928 2.2796 −0.2580 −0.6215 −0.0062 −0.0041 −0.0048 −0.3763 0.0512 −0.1428   0.792 0.443 0.657
    """)
