CFDataRef ff_videotoolbox_hvcc_extradata_create(AVCodecContext *avctx)

{

    HEVCContext *h = avctx->priv_data;

    const HEVCVPS *vps = (const HEVCVPS *)h->ps.vps_list[0]->data;

    const HEVCSPS *sps = (const HEVCSPS *)h->ps.sps_list[0]->data;

    int i, num_pps = 0;

    const HEVCPPS *pps = h->ps.pps;

    PTLCommon ptlc = vps->ptl.general_ptl;

    VUI vui = sps->vui;

    uint8_t parallelismType;

    CFDataRef data = NULL;

    uint8_t *p;

    int vt_extradata_size = 23 + 5 + vps->data_size + 5 + sps->data_size + 3;

    uint8_t *vt_extradata;



    for (i = 0; i < MAX_PPS_COUNT; i++) {

        if (h->ps.pps_list[i]) {

            const HEVCPPS *pps = (const HEVCPPS *)h->ps.pps_list[i]->data;

            vt_extradata_size += 2 + pps->data_size;

            num_pps++;

        }

    }



    vt_extradata = av_malloc(vt_extradata_size);

    if (!vt_extradata)

        return NULL;

    p = vt_extradata;



    /* unsigned int(8) configurationVersion = 1; */

    AV_W8(p + 0, 1);



    /*

     * unsigned int(2) general_profile_space;

     * unsigned int(1) general_tier_flag;

     * unsigned int(5) general_profile_idc;

     */

    AV_W8(p + 1, ptlc.profile_space << 6 |

                 ptlc.tier_flag     << 5 |

                 ptlc.profile_idc);



    /* unsigned int(32) general_profile_compatibility_flags; */

    memcpy(p + 2, ptlc.profile_compatibility_flag, 4);



    /* unsigned int(48) general_constraint_indicator_flags; */

    AV_W8(p + 6, ptlc.progressive_source_flag    << 7 |

                 ptlc.interlaced_source_flag     << 6 |

                 ptlc.non_packed_constraint_flag << 5 |

                 ptlc.frame_only_constraint_flag << 4);

    AV_W8(p + 7, 0);

    AV_WN32(p + 8, 0);



    /* unsigned int(8) general_level_idc; */

    AV_W8(p + 12, ptlc.level_idc);



    /*

     * bit(4) reserved = ‘1111’b;

     * unsigned int(12) min_spatial_segmentation_idc;

     */

    AV_W8(p + 13, 0xf0 | (vui.min_spatial_segmentation_idc >> 4));

    AV_W8(p + 14, vui.min_spatial_segmentation_idc & 0xff);



    /*

     * bit(6) reserved = ‘111111’b;

     * unsigned int(2) parallelismType;

     */

    if (!vui.min_spatial_segmentation_idc)

        parallelismType = 0;

    else if (pps->entropy_coding_sync_enabled_flag && pps->tiles_enabled_flag)

        parallelismType = 0;

    else if (pps->entropy_coding_sync_enabled_flag)

        parallelismType = 3;

    else if (pps->tiles_enabled_flag)

        parallelismType = 2;

    else

        parallelismType = 1;

    AV_W8(p + 15, 0xfc | parallelismType);



    /*

     * bit(6) reserved = ‘111111’b;

     * unsigned int(2) chromaFormat;

     */

    AV_W8(p + 16, sps->chroma_format_idc | 0xfc);



    /*

     * bit(5) reserved = ‘11111’b;

     * unsigned int(3) bitDepthLumaMinus8;

     */

    AV_W8(p + 17, (sps->bit_depth - 8) | 0xfc);



    /*

     * bit(5) reserved = ‘11111’b;

     * unsigned int(3) bitDepthChromaMinus8;

     */

    AV_W8(p + 18, (sps->bit_depth_chroma - 8) | 0xfc);



    /* bit(16) avgFrameRate; */

    AV_WB16(p + 19, 0);



    /*

     * bit(2) constantFrameRate;

     * bit(3) numTemporalLayers;

     * bit(1) temporalIdNested;

     * unsigned int(2) lengthSizeMinusOne;

     */

    AV_W8(p + 21, 0                             << 6 |

                  sps->max_sub_layers           << 3 |

                  sps->temporal_id_nesting_flag << 2 |

                  3);



    /* unsigned int(8) numOfArrays; */

    AV_W8(p + 22, 3);



    p += 23;

    /* vps */

    /*

     * bit(1) array_completeness;

     * unsigned int(1) reserved = 0;

     * unsigned int(6) NAL_unit_type;

     */

    AV_W8(p, 1 << 7 |

             HEVC_NAL_VPS & 0x3f);

    /* unsigned int(16) numNalus; */

    AV_WB16(p + 1, 1);

    /* unsigned int(16) nalUnitLength; */

    AV_WB16(p + 3, vps->data_size);

    /* bit(8*nalUnitLength) nalUnit; */

    memcpy(p + 5, vps->data, vps->data_size);

    p += 5 + vps->data_size;



    /* sps */

    AV_W8(p, 1 << 7 |

             HEVC_NAL_SPS & 0x3f);

    AV_WB16(p + 1, 1);

    AV_WB16(p + 3, sps->data_size);

    memcpy(p + 5, sps->data, sps->data_size);

    p += 5 + sps->data_size;



    /* pps */

    AV_W8(p, 1 << 7 |

             HEVC_NAL_PPS & 0x3f);

    AV_WB16(p + 1, num_pps);

    p += 3;

    for (i = 0; i < MAX_PPS_COUNT; i++) {

        if (h->ps.pps_list[i]) {

            const HEVCPPS *pps = (const HEVCPPS *)h->ps.pps_list[i]->data;

            AV_WB16(p, pps->data_size);

            memcpy(p + 2, pps->data, pps->data_size);

            p += 2 + pps->data_size;

        }

    }



    av_assert0(p - vt_extradata == vt_extradata_size);



    data = CFDataCreate(kCFAllocatorDefault, vt_extradata, vt_extradata_size);

    av_free(vt_extradata);

    return data;

}
