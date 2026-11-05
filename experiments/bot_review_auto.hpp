#pragma once

// Throwaway target for modmesh-bot's auto-path end-to-end test.
// On first APPROVED review the bot runs its configured reviewer
// (`claude -p` wrapper here) and posts the output as a PR comment.

#include <cstring>
#include <string>

namespace experiments
{

// Returns true iff `s` ends with `suffix`. Case-sensitive.
inline bool ends_with(const char * s, const char * suffix)
{
    const size_t ls = std::strlen(s);
    const size_t lsuf = std::strlen(suffix);
    if (lsuf > ls) { return false; }
    return std::strcmp(s + ls - lsuf, suffix) == 0;
}

// Concatenates `n` copies of `unit`. n must be non-negative.
inline std::string repeat(const std::string & unit, int n)
{
    std::string out;
    for (int i = 0; i < n; ++i)
    {
        out += unit;
    }
    return out;
}

} // namespace experiments
