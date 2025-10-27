static bool pte32_match(target_ulong pte0, target_ulong pte1,

                        bool secondary, target_ulong ptem)

{

    return (pte0 & HPTE32_V_VALID)

        && (secondary == !!(pte0 & HPTE32_V_SECONDARY))

        && HPTE32_V_COMPARE(pte0, ptem);

}
