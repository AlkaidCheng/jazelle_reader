/**
 * @file PHKELID.cpp
 * @brief Implementation of the PHKELID bank read method. Store for calorimetry hypothesis.
 * * The PHKELID family contains electron identification data, matching tracks to LAC 
 * energy clusters and testing the electromagnetic shower hypothesis.
 */

#include "jazelle/banks/PHKELID.hpp"
#include "DataBuffer.hpp" // Internal buffer header
#include "jazelle/JazelleEvent.hpp" // For findPHCHRG

namespace jazelle
{
    int32_t PHKELID::read(const DataBuffer& buffer, int32_t offset, JazelleEvent& event)
    {
        int32_t o = offset;
        
        // Read the pointer ID first
        // Bit-shifted value used to link this hypothesis to a specific PHCHRG track ID
        int32_t pid = buffer.readInt(o);
        m_phchrg_id = pid >> 16;
        phchrg = event.get<PHCHRG>().find(m_phchrg_id);
        o += 4;
        
        /* Identification Status and Kinematics */
        idstat  = buffer.readShort(o); o += 2; // Electron ID status flags
        prob    = buffer.readShort(o); o += 2; // Probability/Confidence level of electron hypothesis
        phi     = buffer.readFloat(o); o += 4; // Measured azimuthal angle (phi) of the cluster
        theta   = buffer.readFloat(o); o += 4; // Measured polar angle (theta) of the cluster
        qp      = buffer.readFloat(o); o += 4; // Associated track charge/momentum (q/p)
        dphi    = buffer.readFloat(o); o += 4; // Difference in phi between track and cluster
        dtheta  = buffer.readFloat(o); o += 4; // Difference in theta between track and cluster
        dqp     = buffer.readFloat(o); o += 4; // Error or difference in the q/p measurement
        tphi    = buffer.readFloat(o); o += 4; // Expected phi based on track extrapolation
        ttheta  = buffer.readFloat(o); o += 4; // Expected theta based on track extrapolation
        
        /* Energy and Isolation */
        isolat  = buffer.readFloat(o); o += 4; // Isolation energy around the cluster
        em1     = buffer.readFloat(o); o += 4; // Energy deposited in LAC EM Layer 1
        em12    = buffer.readFloat(o); o += 4; // Combined energy in EM Layers 1 and 2
        dem12   = buffer.readFloat(o); o += 4; // Error/uncertainty in the combined EM energy
        had1    = buffer.readFloat(o); o += 4; // Energy in LAC Hadronic Layer 1 (used for rejection)
        
        /* Shower Shape Parameters */
        emphi   = buffer.readFloat(o); o += 4; // EM shower centroid phi
        emtheta = buffer.readFloat(o); o += 4; // EM shower centroid theta
        phiwid  = buffer.readFloat(o); o += 4; // Width of the EM shower in phi
        thewid  = buffer.readFloat(o); o += 4; // Width of the EM shower in theta
        
        /* Transverse Energy Profiles (Tower Grids) */
        // Energy sums in varying tower grid sizes around the track impact point
        em1x1   = buffer.readFloat(o); o += 4; // Energy in a 1x1 tower window
        em2x2a  = buffer.readFloat(o); o += 4; // Energy in a 2x2 tower window (offset A)
        em2x2b  = buffer.readFloat(o); o += 4; // Energy in a 2x2 tower window (offset B)
        em3x3a  = buffer.readFloat(o); o += 4; // Energy in a 3x3 tower window (offset A)
        em3x3b  = buffer.readFloat(o); o += 4; // Energy in a 3x3 tower window (offset B)
        
        return 96; // Fixed size
    }

} // namespace jazelle