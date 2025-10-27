static uint64_t vtd_get_iotlb_key(uint64_t gfn, uint8_t source_id,

                                  uint32_t level)

{

    return gfn | ((uint64_t)(source_id) << VTD_IOTLB_SID_SHIFT) |

           ((uint64_t)(level) << VTD_IOTLB_LVL_SHIFT);

}
