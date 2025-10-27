static VTDIOTLBEntry *vtd_lookup_iotlb(IntelIOMMUState *s, uint16_t source_id,

                                       hwaddr addr)

{

    uint64_t key;



    key = (addr >> VTD_PAGE_SHIFT_4K) |

           ((uint64_t)(source_id) << VTD_IOTLB_SID_SHIFT);

    return g_hash_table_lookup(s->iotlb, &key);



}
