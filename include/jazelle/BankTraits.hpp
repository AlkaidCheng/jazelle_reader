#pragma once
#include <string_view>
#include "banks/AllBanks.hpp"

namespace jazelle
{
    template <typename T> inline constexpr std::string_view bank_name = "UNKNOWN";
    
    template<> inline constexpr std::string_view bank_name<MCHEAD>  = "MCHEAD";
    template<> inline constexpr std::string_view bank_name<MCPART>  = "MCPART";
    template<> inline constexpr std::string_view bank_name<PHPSUM>  = "PHPSUM";
    template<> inline constexpr std::string_view bank_name<PHCHRG>  = "PHCHRG";
    template<> inline constexpr std::string_view bank_name<PHKLUS>  = "PHKLUS";
    template<> inline constexpr std::string_view bank_name<PHWIC>   = "PHWIC";
    template<> inline constexpr std::string_view bank_name<PHCRID>  = "PHCRID";
    template<> inline constexpr std::string_view bank_name<PHKTRK>  = "PHKTRK";
    template<> inline constexpr std::string_view bank_name<PHKELID> = "PHKELID";
}