/**
 * @file PHCHRG.hpp
 * @brief Definition of the PHCHRG (Charged Particle) bank.
 *
 * Contains the reconstructed parameters for a charged track, combining data 
 * from the Central Drift Chamber (CDC) and Vertex Detector (VXD).
 */

#pragma once

#include "../Bank.hpp"
#include <cstdint>
#include <array>

namespace jazelle
{
    class DataBuffer;
    class JazelleEvent;

    struct PHCHRG : public Bank
    {
        /**
         * @brief Track helix parameters at origin.
         * * The 6 elements correspond to:
         * - hlxpar[0] = phi
         * - hlxpar[1] = 1 / p_T (inverse transverse momentum)
         * - hlxpar[2] = tan(lambda) (dip angle)
         * - hlxpar[3] = x
         * - hlxpar[4] = y
         * - hlxpar[5] = z
         * * Momentum conversion formulas:
         * $$p_T = \frac{1}{\text{hlxpar}[1]}$$
         * $$p_x = p_T \cos(\text{hlxpar}[0])$$
         * $$p_y = p_T \sin(\text{hlxpar}[0])$$
         * $$p_z = p_T \text{hlxpar}[2]$$
         */
        std::array<float, 6>  hlxpar;   
        
        /**
         * @brief 5x5 symmetric error matrix for the helix parameters.
         * * Note: While hlxpar provides (x, y, z) for convenience, the error matrix 
         * corresponds to the 5 local fitting parameters: (phi, 1/pt, tan(lambda), TSI, ETA).
         * * TSI and ETA are distances to a given point defined by the magnetic field and momentum vectors:
         * - TSI = TN \cdot \Delta
         * - ETA = TQ \cdot \Delta
         * (Where \Delta = (x-x0, y-y0, z-z0), TN is the normal vector, and TQ is the binormal vector).
         */
        std::array<float, 15> dhlxpar;  
        
        float   bnorm;    ///< 2D Impact parameter (distance of closest approach to the Z axis)
        float   impact;   ///< Error/Significance of the 2D impact parameter
        float   b3norm;   ///< 3D Impact parameter (closest approach to the 3D primary vertex)
        float   impact3;  ///< Error/Significance of the 3D impact parameter
        
        int16_t charge;   ///< Reconstructed charge (+1 or -1)
        int16_t smwstat;  ///< Status of track swimming/extrapolation through the magnetic field
        int32_t status;   ///< General track quality and reconstruction status mask
        
        float   tkpar0;   ///< Reference point parameter for the track fit
        std::array<float, 5>  tkpar;    ///< Alternative track fit parameters (e.g., at calorimeter face)
        std::array<float, 15> dtkpar;   ///< Error matrix for alternative track parameters
        
        float   length;   ///< Total reconstructed arc length of the track
        float   chi2dt;   ///< Chi-squared of the tracking fit
        int16_t imc;      ///< Pointer/Match to corresponding MCPART (if analyzing Monte Carlo)
        int16_t ndfdt;    ///< Degrees of freedom for the tracking fit
        
        // --- Hit Counting ---
        int16_t nhit;     ///< Total number of hits on the track
        int16_t nhite;    ///< Number of expected hits based on trajectory
        int16_t nhitp;    ///< Number of hits actually used in the final fit
        int16_t nmisht;   ///< Number of missing/dropped hits
        int16_t nwrght;   ///< Number of wrong/noise hits assigned to the track
        int16_t nhitv;    ///< Number of hits specifically in the Vertex Detector (VXD)
        
        float   chi2;     ///< Overall chi-squared of the track
        float   chi2v;    ///< Chi-squared contribution specifically from the VXD hits
        int32_t vxdhit;   ///< Bitmask indicating which specific VXD layers were hit
        
        int16_t mustat;   ///< Muon system (WIC) matching status
        int16_t estat;    ///< Electron/Calorimeter matching status
        int32_t dedx;     ///< Encoded dE/dx (ionization energy loss) used for particle ID

        explicit PHCHRG(int32_t id) : Bank(id) {}

        int32_t read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event) override;
    };
} // namespace jazelle