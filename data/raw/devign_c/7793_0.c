static uint64_t vtd_context_cache_invalidate(IntelIOMMUState *s, uint64_t val)

{

    uint64_t caig;

    uint64_t type = val & VTD_CCMD_CIRG_MASK;



    switch (type) {

    case VTD_CCMD_GLOBAL_INVL:

        VTD_DPRINTF(INV, "Global invalidation request");

        caig = VTD_CCMD_GLOBAL_INVL_A;

        break;



    case VTD_CCMD_DOMAIN_INVL:

        VTD_DPRINTF(INV, "Domain-selective invalidation request");

        caig = VTD_CCMD_DOMAIN_INVL_A;

        break;



    case VTD_CCMD_DEVICE_INVL:

        VTD_DPRINTF(INV, "Domain-selective invalidation request");

        caig = VTD_CCMD_DEVICE_INVL_A;

        break;



    default:

        VTD_DPRINTF(GENERAL,

                    "error: wrong context-cache invalidation granularity");

        caig = 0;

    }

    return caig;

}
