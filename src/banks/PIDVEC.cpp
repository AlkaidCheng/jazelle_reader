/**
 * @file PIDVEC.cpp
 * @brief Implementation of the PIDVEC helper struct.
 */

#include "jazelle/banks/PIDVEC.hpp"
#include "DataBuffer.hpp" // Internal buffer header

namespace jazelle
{
    PIDVEC::PIDVEC(const DataBuffer& data, int32_t offset)
    {
        e  = data.readFloat(offset);
        mu = data.readFloat(offset + 4);
        pi = data.readFloat(offset + 8);
        k  = data.readFloat(offset + 12);
        p  = data.readFloat(offset + 16);
    }

    PIDVEC::PIDVEC(const PIDVEC* liq, const PIDVEC* gas, float norm)
    {
        e  = norm;
        mu = norm;
        pi = norm;
        k  = norm;
        p  = norm;
        
        if (liq)
        {
            e  += liq->e;
            mu += liq->mu;
            pi += liq->pi;
            k  += liq->k;
            p  += liq->p;
        }
        if (gas)
        {
            e  += gas->e;
            mu += gas->mu;
            pi += gas->pi;
            k  += gas->k;
            p  += gas->p;
        }
    }

} // namespace jazelle