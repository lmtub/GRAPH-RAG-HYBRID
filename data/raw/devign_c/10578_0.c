static void vtd_context_device_invalidate(IntelIOMMUState *s,

                                          uint16_t source_id,

                                          uint16_t func_mask)

{

    uint16_t mask;

    VTDAddressSpace **pvtd_as;

    VTDAddressSpace *vtd_as;

    uint16_t devfn;

    uint16_t devfn_it;



    switch (func_mask & 3) {

    case 0:

        mask = 0;   /* No bits in the SID field masked */

        break;

    case 1:

        mask = 4;   /* Mask bit 2 in the SID field */

        break;

    case 2:

        mask = 6;   /* Mask bit 2:1 in the SID field */

        break;

    case 3:

        mask = 7;   /* Mask bit 2:0 in the SID field */

        break;

    }

    VTD_DPRINTF(INV, "device-selective invalidation source 0x%"PRIx16

                    " mask %"PRIu16, source_id, mask);

    pvtd_as = s->address_spaces[VTD_SID_TO_BUS(source_id)];

    if (pvtd_as) {

        devfn = VTD_SID_TO_DEVFN(source_id);

        for (devfn_it = 0; devfn_it < VTD_PCI_DEVFN_MAX; ++devfn_it) {

            vtd_as = pvtd_as[devfn_it];

            if (vtd_as && ((devfn_it & mask) == (devfn & mask))) {

                VTD_DPRINTF(INV, "invalidate context-cahce of devfn 0x%"PRIx16,

                            devfn_it);

                vtd_as->context_cache_entry.context_cache_gen = 0;

            }

        }

    }

}
