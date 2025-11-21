#pragma once
#include <string_view>
#include "banks/AllBanks.hpp"

namespace jazelle
{
    template <typename T> constexpr std::string_view bank_name = "UNKNOWN";
    
    template<> constexpr std::string_view bank_name<MCHEAD>  = "MCHEAD";
    template<> constexpr std::string_view bank_name<MCPART>  = "MCPART";
    template<> constexpr std::string_view bank_name<PHPSUM>  = "PHPSUM";
    template<> constexpr std::string_view bank_name<PHCHRG>  = "PHCHRG";
    template<> constexpr std::string_view bank_name<PHKLUS>  = "PHKLUS";
    template<> constexpr std::string_view bank_name<PHWIC>   = "PHWIC";
    template<> constexpr std::string_view bank_name<PHCRID>  = "PHCRID";
    template<> constexpr std::string_view bank_name<PHKTRK>  = "PHKTRK";
    template<> constexpr std::string_view bank_name<PHKELID> = "PHKELID";
}