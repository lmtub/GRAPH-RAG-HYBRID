static void iv_Decode_Chunk(Indeo3DecodeContext *s,

        uint8_t *cur, uint8_t *ref, int width, int height,

        const uint8_t *buf1, int cb_offset, const uint8_t *hdr,

        const uint8_t *buf2, int min_width_160)

{

    uint8_t bit_buf;

    unsigned int bit_pos, lv, lv1, lv2;

    int *width_tbl, width_tbl_arr[10];

    const signed char *ref_vectors;

    uint8_t *cur_frm_pos, *ref_frm_pos, *cp, *cp2;


    uint32_t *cur_lp, *ref_lp;

    const uint32_t *correction_lp[2], *correctionloworder_lp[2], *correctionhighorder_lp[2];

    uint8_t *correction_type_sp[2];

    struct ustr strip_tbl[20], *strip;

    int i, j, k, lp1, lp2, flag1, cmd, blks_width, blks_height, region_160_width,

        rle_v1, rle_v2, rle_v3;

    unsigned short res;



    bit_buf = 0;

    ref_vectors = NULL;



    width_tbl = width_tbl_arr + 1;

    i = (width < 0 ? width + 3 : width)/4;

    for(j = -1; j < 8; j++)

        width_tbl[j] = i * j;



    strip = strip_tbl;



    for(region_160_width = 0; region_160_width < (width - min_width_160); region_160_width += min_width_160);



    strip->ypos = strip->xpos = 0;

    for(strip->width = min_width_160; width > strip->width; strip->width *= 2);

    strip->height = height;

    strip->split_direction = 0;

    strip->split_flag = 0;

    strip->usl7 = 0;



    bit_pos = 0;



    rle_v1 = rle_v2 = rle_v3 = 0;



    while(strip >= strip_tbl) {

        if(bit_pos <= 0) {

            bit_pos = 8;

            bit_buf = *buf1++;

        }



        bit_pos -= 2;

        cmd = (bit_buf >> bit_pos) & 0x03;



        if(cmd == 0) {

            strip++;

            if(strip >= strip_tbl + FF_ARRAY_ELEMS(strip_tbl)) {

                av_log(s->avctx, AV_LOG_WARNING, "out of range strip\n");

                break;

            }

            memcpy(strip, strip-1, sizeof(*strip));

            strip->split_flag = 1;

            strip->split_direction = 0;

            strip->height = (strip->height > 8 ? ((strip->height+8)>>4)<<3 : 4);

            continue;

        } else if(cmd == 1) {

            strip++;

            if(strip >= strip_tbl + FF_ARRAY_ELEMS(strip_tbl)) {

                av_log(s->avctx, AV_LOG_WARNING, "out of range strip\n");

                break;

            }

            memcpy(strip, strip-1, sizeof(*strip));

            strip->split_flag = 1;

            strip->split_direction = 1;

            strip->width = (strip->width > 8 ? ((strip->width+8)>>4)<<3 : 4);

            continue;

        } else if(cmd == 2) {

            if(strip->usl7 == 0) {

                strip->usl7 = 1;

                ref_vectors = NULL;

                continue;

            }

        } else if(cmd == 3) {

            if(strip->usl7 == 0) {

                strip->usl7 = 1;

                ref_vectors = (const signed char*)buf2 + (*buf1 * 2);

                buf1++;

                continue;

            }

        }



        cur_frm_pos = cur + width * strip->ypos + strip->xpos;



        if((blks_width = strip->width) < 0)

            blks_width += 3;

        blks_width >>= 2;

        blks_height = strip->height;



        if(ref_vectors != NULL) {

            ref_frm_pos = ref + (ref_vectors[0] + strip->ypos) * width +

                ref_vectors[1] + strip->xpos;

        } else

            ref_frm_pos = cur_frm_pos - width_tbl[4];



        if(cmd == 2) {

            if(bit_pos <= 0) {

                bit_pos = 8;

                bit_buf = *buf1++;

            }



            bit_pos -= 2;

            cmd = (bit_buf >> bit_pos) & 0x03;



            if(cmd == 0 || ref_vectors != NULL) {

                for(lp1 = 0; lp1 < blks_width; lp1++) {

                    for(i = 0, j = 0; i < blks_height; i++, j += width_tbl[1])

                        ((uint32_t *)cur_frm_pos)[j] = ((uint32_t *)ref_frm_pos)[j];

                    cur_frm_pos += 4;

                    ref_frm_pos += 4;

                }

            } else if(cmd != 1)

                return;

        } else {

            k = *buf1 >> 4;

            j = *buf1 & 0x0f;

            buf1++;

            lv = j + cb_offset;



            if((lv - 8) <= 7 && (k == 0 || k == 3 || k == 10)) {

                cp2 = s->ModPred + ((lv - 8) << 7);

                cp = ref_frm_pos;

                for(i = 0; i < blks_width << 2; i++) {

                    int v = *cp >> 1;

                    *(cp++) = cp2[v];

                }

            }



            if(k == 1 || k == 4) {

                lv = (hdr[j] & 0xf) + cb_offset;

                correction_type_sp[0] = s->corrector_type + (lv << 8);

                correction_lp[0] = correction + (lv << 8);

                lv = (hdr[j] >> 4) + cb_offset;

                correction_lp[1] = correction + (lv << 8);

                correction_type_sp[1] = s->corrector_type + (lv << 8);

            } else {

                correctionloworder_lp[0] = correctionloworder_lp[1] = correctionloworder + (lv << 8);

                correctionhighorder_lp[0] = correctionhighorder_lp[1] = correctionhighorder + (lv << 8);

                correction_type_sp[0] = correction_type_sp[1] = s->corrector_type + (lv << 8);

                correction_lp[0] = correction_lp[1] = correction + (lv << 8);

            }



            switch(k) {

            case 1:

            case 0:                    /********** CASE 0 **********/

                for( ; blks_height > 0; blks_height -= 4) {

                    for(lp1 = 0; lp1 < blks_width; lp1++) {

                        for(lp2 = 0; lp2 < 4; ) {

                            k = *buf1++;

                            cur_lp = ((uint32_t *)cur_frm_pos) + width_tbl[lp2];

                            ref_lp = ((uint32_t *)ref_frm_pos) + width_tbl[lp2];

                            if ((uint8_t *)cur_lp >= cur_end-3)

                                break;



                            switch(correction_type_sp[0][k]) {

                            case 0:

                                *cur_lp = av_le2ne32(((av_le2ne32(*ref_lp) >> 1) + correction_lp[lp2 & 0x01][k]) << 1);

                                lp2++;

                                break;

                            case 1:

                                res = ((av_le2ne16(((unsigned short *)(ref_lp))[0]) >> 1) + correction_lp[lp2 & 0x01][*buf1]) << 1;

                                ((unsigned short *)cur_lp)[0] = av_le2ne16(res);

                                res = ((av_le2ne16(((unsigned short *)(ref_lp))[1]) >> 1) + correction_lp[lp2 & 0x01][k]) << 1;

                                ((unsigned short *)cur_lp)[1] = av_le2ne16(res);

                                buf1++;

                                lp2++;

                                break;

                            case 2:

                                if(lp2 == 0) {

                                    for(i = 0, j = 0; i < 2; i++, j += width_tbl[1])

                                        cur_lp[j] = ref_lp[j];

                                    lp2 += 2;

                                }

                                break;

                            case 3:

                                if(lp2 < 2) {

                                    for(i = 0, j = 0; i < (3 - lp2); i++, j += width_tbl[1])

                                        cur_lp[j] = ref_lp[j];

                                    lp2 = 3;

                                }

                                break;

                            case 8:

                                if(lp2 == 0) {

                                    RLE_V3_CHECK(buf1,rle_v1,rle_v2,rle_v3)



                                    if(rle_v1 == 1 || ref_vectors != NULL) {

                                        for(i = 0, j = 0; i < 4; i++, j += width_tbl[1])

                                            cur_lp[j] = ref_lp[j];

                                    }



                                    RLE_V2_CHECK(buf1,rle_v2, rle_v3,lp2)

                                    break;

                                } else {

                                    rle_v1 = 1;

                                    rle_v2 = *buf1 - 1;

                                }

                            case 5:

                                LP2_CHECK(buf1,rle_v3,lp2)

                            case 4:

                                for(i = 0, j = 0; i < (4 - lp2); i++, j += width_tbl[1])

                                    cur_lp[j] = ref_lp[j];

                                lp2 = 4;

                                break;



                            case 7:

                                if(rle_v3 != 0)

                                    rle_v3 = 0;

                                else {

                                    buf1--;

                                    rle_v3 = 1;

                                }

                            case 6:

                                if(ref_vectors != NULL) {

                                    for(i = 0, j = 0; i < 4; i++, j += width_tbl[1])

                                        cur_lp[j] = ref_lp[j];

                                }

                                lp2 = 4;

                                break;



                            case 9:

                                lv1 = *buf1++;

                                lv = (lv1 & 0x7F) << 1;

                                lv += (lv << 8);

                                lv += (lv << 16);

                                for(i = 0, j = 0; i < 4; i++, j += width_tbl[1])

                                    cur_lp[j] = lv;



                                LV1_CHECK(buf1,rle_v3,lv1,lp2)

                                break;

                            default:

                                return;

                            }

                        }



                        cur_frm_pos += 4;

                        ref_frm_pos += 4;

                    }



                    cur_frm_pos += ((width - blks_width) * 4);

                    ref_frm_pos += ((width - blks_width) * 4);

                }

                break;



            case 4:

            case 3:                    /********** CASE 3 **********/

                if(ref_vectors != NULL)

                    return;

                flag1 = 1;



                for( ; blks_height > 0; blks_height -= 8) {

                    for(lp1 = 0; lp1 < blks_width; lp1++) {

                        for(lp2 = 0; lp2 < 4; ) {

                            k = *buf1++;



                            cur_lp = ((uint32_t *)cur_frm_pos) + width_tbl[lp2 * 2];

                            ref_lp = ((uint32_t *)cur_frm_pos) + width_tbl[(lp2 * 2) - 1];



                            switch(correction_type_sp[lp2 & 0x01][k]) {

                            case 0:

                                cur_lp[width_tbl[1]] = av_le2ne32(((av_le2ne32(*ref_lp) >> 1) + correction_lp[lp2 & 0x01][k]) << 1);

                                if(lp2 > 0 || flag1 == 0 || strip->ypos != 0)

                                    cur_lp[0] = ((cur_lp[-width_tbl[1]] >> 1) + (cur_lp[width_tbl[1]] >> 1)) & 0xFEFEFEFE;

                                else

                                    cur_lp[0] = av_le2ne32(((av_le2ne32(*ref_lp) >> 1) + correction_lp[lp2 & 0x01][k]) << 1);

                                lp2++;

                                break;



                            case 1:

                                res = ((av_le2ne16(((unsigned short *)ref_lp)[0]) >> 1) + correction_lp[lp2 & 0x01][*buf1]) << 1;

                                ((unsigned short *)cur_lp)[width_tbl[2]] = av_le2ne16(res);

                                res = ((av_le2ne16(((unsigned short *)ref_lp)[1]) >> 1) + correction_lp[lp2 & 0x01][k]) << 1;

                                ((unsigned short *)cur_lp)[width_tbl[2]+1] = av_le2ne16(res);



                                if(lp2 > 0 || flag1 == 0 || strip->ypos != 0)

                                    cur_lp[0] = ((cur_lp[-width_tbl[1]] >> 1) + (cur_lp[width_tbl[1]] >> 1)) & 0xFEFEFEFE;

                                else

                                    cur_lp[0] = cur_lp[width_tbl[1]];

                                buf1++;

                                lp2++;

                                break;



                            case 2:

                                if(lp2 == 0) {

                                    for(i = 0, j = 0; i < 4; i++, j += width_tbl[1])

                                        cur_lp[j] = *ref_lp;

                                    lp2 += 2;

                                }

                                break;



                            case 3:

                                if(lp2 < 2) {

                                    for(i = 0, j = 0; i < 6 - (lp2 * 2); i++, j += width_tbl[1])

                                        cur_lp[j] = *ref_lp;

                                    lp2 = 3;

                                }

                                break;



                            case 6:

                                lp2 = 4;

                                break;



                            case 7:

                                if(rle_v3 != 0)

                                    rle_v3 = 0;

                                else {

                                    buf1--;

                                    rle_v3 = 1;

                                }

                                lp2 = 4;

                                break;



                            case 8:

                                if(lp2 == 0) {

                                    RLE_V3_CHECK(buf1,rle_v1,rle_v2,rle_v3)



                                    if(rle_v1 == 1) {

                                        for(i = 0, j = 0; i < 8; i++, j += width_tbl[1])

                                            cur_lp[j] = ref_lp[j];

                                    }



                                    RLE_V2_CHECK(buf1,rle_v2, rle_v3,lp2)

                                    break;

                                } else {

                                    rle_v2 = (*buf1) - 1;

                                    rle_v1 = 1;

                                }

                            case 5:

                                LP2_CHECK(buf1,rle_v3,lp2)

                            case 4:

                                for(i = 0, j = 0; i < 8 - (lp2 * 2); i++, j += width_tbl[1])

                                    cur_lp[j] = *ref_lp;

                                lp2 = 4;

                                break;



                            case 9:

                                av_log(s->avctx, AV_LOG_ERROR, "UNTESTED.\n");

                                lv1 = *buf1++;

                                lv = (lv1 & 0x7F) << 1;

                                lv += (lv << 8);

                                lv += (lv << 16);



                                for(i = 0, j = 0; i < 4; i++, j += width_tbl[1])

                                    cur_lp[j] = lv;



                                LV1_CHECK(buf1,rle_v3,lv1,lp2)

                                break;



                            default:

                                return;

                            }

                        }



                        cur_frm_pos += 4;

                    }



                    cur_frm_pos += (((width * 2) - blks_width) * 4);

                    flag1 = 0;

                }

                break;



            case 10:                    /********** CASE 10 **********/

                if(ref_vectors == NULL) {

                    flag1 = 1;



                    for( ; blks_height > 0; blks_height -= 8) {

                        for(lp1 = 0; lp1 < blks_width; lp1 += 2) {

                            for(lp2 = 0; lp2 < 4; ) {

                                k = *buf1++;

                                cur_lp = ((uint32_t *)cur_frm_pos) + width_tbl[lp2 * 2];

                                ref_lp = ((uint32_t *)cur_frm_pos) + width_tbl[(lp2 * 2) - 1];

                                lv1 = ref_lp[0];

                                lv2 = ref_lp[1];

                                if(lp2 == 0 && flag1 != 0) {

#if HAVE_BIGENDIAN

                                    lv1 = lv1 & 0xFF00FF00;

                                    lv1 = (lv1 >> 8) | lv1;

                                    lv2 = lv2 & 0xFF00FF00;

                                    lv2 = (lv2 >> 8) | lv2;

#else

                                    lv1 = lv1 & 0x00FF00FF;

                                    lv1 = (lv1 << 8) | lv1;

                                    lv2 = lv2 & 0x00FF00FF;

                                    lv2 = (lv2 << 8) | lv2;

#endif

                                }



                                switch(correction_type_sp[lp2 & 0x01][k]) {

                                case 0:

                                    cur_lp[width_tbl[1]] = av_le2ne32(((av_le2ne32(lv1) >> 1) + correctionloworder_lp[lp2 & 0x01][k]) << 1);

                                    cur_lp[width_tbl[1]+1] = av_le2ne32(((av_le2ne32(lv2) >> 1) + correctionhighorder_lp[lp2 & 0x01][k]) << 1);

                                    if(lp2 > 0 || strip->ypos != 0 || flag1 == 0) {

                                        cur_lp[0] = ((cur_lp[-width_tbl[1]] >> 1) + (cur_lp[width_tbl[1]] >> 1)) & 0xFEFEFEFE;

                                        cur_lp[1] = ((cur_lp[-width_tbl[1]+1] >> 1) + (cur_lp[width_tbl[1]+1] >> 1)) & 0xFEFEFEFE;

                                    } else {

                                        cur_lp[0] = cur_lp[width_tbl[1]];

                                        cur_lp[1] = cur_lp[width_tbl[1]+1];

                                    }

                                    lp2++;

                                    break;



                                case 1:

                                    cur_lp[width_tbl[1]] = av_le2ne32(((av_le2ne32(lv1) >> 1) + correctionloworder_lp[lp2 & 0x01][*buf1]) << 1);

                                    cur_lp[width_tbl[1]+1] = av_le2ne32(((av_le2ne32(lv2) >> 1) + correctionloworder_lp[lp2 & 0x01][k]) << 1);

                                    if(lp2 > 0 || strip->ypos != 0 || flag1 == 0) {

                                        cur_lp[0] = ((cur_lp[-width_tbl[1]] >> 1) + (cur_lp[width_tbl[1]] >> 1)) & 0xFEFEFEFE;

                                        cur_lp[1] = ((cur_lp[-width_tbl[1]+1] >> 1) + (cur_lp[width_tbl[1]+1] >> 1)) & 0xFEFEFEFE;

                                    } else {

                                        cur_lp[0] = cur_lp[width_tbl[1]];

                                        cur_lp[1] = cur_lp[width_tbl[1]+1];

                                    }

                                    buf1++;

                                    lp2++;

                                    break;



                                case 2:

                                    if(lp2 == 0) {

                                        if(flag1 != 0) {

                                            for(i = 0, j = width_tbl[1]; i < 3; i++, j += width_tbl[1]) {

                                                cur_lp[j] = lv1;

                                                cur_lp[j+1] = lv2;

                                            }

                                            cur_lp[0] = ((cur_lp[-width_tbl[1]] >> 1) + (cur_lp[width_tbl[1]] >> 1)) & 0xFEFEFEFE;

                                            cur_lp[1] = ((cur_lp[-width_tbl[1]+1] >> 1) + (cur_lp[width_tbl[1]+1] >> 1)) & 0xFEFEFEFE;

                                        } else {

                                            for(i = 0, j = 0; i < 4; i++, j += width_tbl[1]) {

                                                cur_lp[j] = lv1;

                                                cur_lp[j+1] = lv2;

                                            }

                                        }

                                        lp2 += 2;

                                    }

                                    break;



                                case 3:

                                    if(lp2 < 2) {

                                        if(lp2 == 0 && flag1 != 0) {

                                            for(i = 0, j = width_tbl[1]; i < 5; i++, j += width_tbl[1]) {

                                                cur_lp[j] = lv1;

                                                cur_lp[j+1] = lv2;

                                            }

                                            cur_lp[0] = ((cur_lp[-width_tbl[1]] >> 1) + (cur_lp[width_tbl[1]] >> 1)) & 0xFEFEFEFE;

                                            cur_lp[1] = ((cur_lp[-width_tbl[1]+1] >> 1) + (cur_lp[width_tbl[1]+1] >> 1)) & 0xFEFEFEFE;

                                        } else {

                                            for(i = 0, j = 0; i < 6 - (lp2 * 2); i++, j += width_tbl[1]) {

                                                cur_lp[j] = lv1;

                                                cur_lp[j+1] = lv2;

                                            }

                                        }

                                        lp2 = 3;

                                    }

                                    break;



                                case 8:

                                    if(lp2 == 0) {

                                        RLE_V3_CHECK(buf1,rle_v1,rle_v2,rle_v3)

                                        if(rle_v1 == 1) {

                                            if(flag1 != 0) {

                                                for(i = 0, j = width_tbl[1]; i < 7; i++, j += width_tbl[1]) {

                                                    cur_lp[j] = lv1;

                                                    cur_lp[j+1] = lv2;

                                                }

                                                cur_lp[0] = ((cur_lp[-width_tbl[1]] >> 1) + (cur_lp[width_tbl[1]] >> 1)) & 0xFEFEFEFE;

                                                cur_lp[1] = ((cur_lp[-width_tbl[1]+1] >> 1) + (cur_lp[width_tbl[1]+1] >> 1)) & 0xFEFEFEFE;

                                            } else {

                                                for(i = 0, j = 0; i < 8; i++, j += width_tbl[1]) {

                                                    cur_lp[j] = lv1;

                                                    cur_lp[j+1] = lv2;

                                                }

                                            }

                                        }

                                        RLE_V2_CHECK(buf1,rle_v2, rle_v3,lp2)

                                        break;

                                    } else {

                                        rle_v1 = 1;

                                        rle_v2 = (*buf1) - 1;

                                    }

                                case 5:

                                    LP2_CHECK(buf1,rle_v3,lp2)

                                case 4:

                                    if(lp2 == 0 && flag1 != 0) {

                                        for(i = 0, j = width_tbl[1]; i < 7; i++, j += width_tbl[1]) {

                                            cur_lp[j] = lv1;

                                            cur_lp[j+1] = lv2;

                                        }

                                        cur_lp[0] = ((cur_lp[-width_tbl[1]] >> 1) + (cur_lp[width_tbl[1]] >> 1)) & 0xFEFEFEFE;

                                        cur_lp[1] = ((cur_lp[-width_tbl[1]+1] >> 1) + (cur_lp[width_tbl[1]+1] >> 1)) & 0xFEFEFEFE;

                                    } else {

                                        for(i = 0, j = 0; i < 8 - (lp2 * 2); i++, j += width_tbl[1]) {

                                            cur_lp[j] = lv1;

                                            cur_lp[j+1] = lv2;

                                        }

                                    }

                                    lp2 = 4;

                                    break;



                                case 6:

                                    lp2 = 4;

                                    break;



                                case 7:

                                    if(lp2 == 0) {

                                        if(rle_v3 != 0)

                                            rle_v3 = 0;

                                        else {

                                            buf1--;

                                            rle_v3 = 1;

                                        }

                                        lp2 = 4;

                                    }

                                    break;



                                case 9:

                                    av_log(s->avctx, AV_LOG_ERROR, "UNTESTED.\n");

                                    lv1 = *buf1;

                                    lv = (lv1 & 0x7F) << 1;

                                    lv += (lv << 8);

                                    lv += (lv << 16);

                                    for(i = 0, j = 0; i < 8; i++, j += width_tbl[1])

                                        cur_lp[j] = lv;

                                    LV1_CHECK(buf1,rle_v3,lv1,lp2)

                                    break;



                                default:

                                    return;

                                }

                            }



                            cur_frm_pos += 8;

                        }



                        cur_frm_pos += (((width * 2) - blks_width) * 4);

                        flag1 = 0;

                    }

                } else {

                    for( ; blks_height > 0; blks_height -= 8) {

                        for(lp1 = 0; lp1 < blks_width; lp1 += 2) {

                            for(lp2 = 0; lp2 < 4; ) {

                                k = *buf1++;

                                cur_lp = ((uint32_t *)cur_frm_pos) + width_tbl[lp2 * 2];

                                ref_lp = ((uint32_t *)ref_frm_pos) + width_tbl[lp2 * 2];



                                switch(correction_type_sp[lp2 & 0x01][k]) {

                                case 0:

                                    lv1 = correctionloworder_lp[lp2 & 0x01][k];

                                    lv2 = correctionhighorder_lp[lp2 & 0x01][k];

                                    cur_lp[0] = av_le2ne32(((av_le2ne32(ref_lp[0]) >> 1) + lv1) << 1);

                                    cur_lp[1] = av_le2ne32(((av_le2ne32(ref_lp[1]) >> 1) + lv2) << 1);

                                    cur_lp[width_tbl[1]] = av_le2ne32(((av_le2ne32(ref_lp[width_tbl[1]]) >> 1) + lv1) << 1);

                                    cur_lp[width_tbl[1]+1] = av_le2ne32(((av_le2ne32(ref_lp[width_tbl[1]+1]) >> 1) + lv2) << 1);

                                    lp2++;

                                    break;



                                case 1:

                                    lv1 = correctionloworder_lp[lp2 & 0x01][*buf1++];

                                    lv2 = correctionloworder_lp[lp2 & 0x01][k];

                                    cur_lp[0] = av_le2ne32(((av_le2ne32(ref_lp[0]) >> 1) + lv1) << 1);

                                    cur_lp[1] = av_le2ne32(((av_le2ne32(ref_lp[1]) >> 1) + lv2) << 1);

                                    cur_lp[width_tbl[1]] = av_le2ne32(((av_le2ne32(ref_lp[width_tbl[1]]) >> 1) + lv1) << 1);

                                    cur_lp[width_tbl[1]+1] = av_le2ne32(((av_le2ne32(ref_lp[width_tbl[1]+1]) >> 1) + lv2) << 1);

                                    lp2++;

                                    break;



                                case 2:

                                    if(lp2 == 0) {

                                        for(i = 0, j = 0; i < 4; i++, j += width_tbl[1]) {

                                            cur_lp[j] = ref_lp[j];

                                            cur_lp[j+1] = ref_lp[j+1];

                                        }

                                        lp2 += 2;

                                    }

                                    break;



                                case 3:

                                    if(lp2 < 2) {

                                        for(i = 0, j = 0; i < 6 - (lp2 * 2); i++, j += width_tbl[1]) {

                                            cur_lp[j] = ref_lp[j];

                                            cur_lp[j+1] = ref_lp[j+1];

                                        }

                                        lp2 = 3;

                                    }

                                    break;



                                case 8:

                                    if(lp2 == 0) {

                                        RLE_V3_CHECK(buf1,rle_v1,rle_v2,rle_v3)

                                        for(i = 0, j = 0; i < 8; i++, j += width_tbl[1]) {

                                            ((uint32_t *)cur_frm_pos)[j] = ((uint32_t *)ref_frm_pos)[j];

                                            ((uint32_t *)cur_frm_pos)[j+1] = ((uint32_t *)ref_frm_pos)[j+1];

                                        }

                                        RLE_V2_CHECK(buf1,rle_v2, rle_v3,lp2)

                                        break;

                                    } else {

                                        rle_v1 = 1;

                                        rle_v2 = (*buf1) - 1;

                                    }

                                case 5:

                                case 7:

                                    LP2_CHECK(buf1,rle_v3,lp2)

                                case 6:

                                case 4:

                                    for(i = 0, j = 0; i < 8 - (lp2 * 2); i++, j += width_tbl[1]) {

                                        cur_lp[j] = ref_lp[j];

                                        cur_lp[j+1] = ref_lp[j+1];

                                    }

                                    lp2 = 4;

                                    break;



                                case 9:

                                    av_log(s->avctx, AV_LOG_ERROR, "UNTESTED.\n");

                                    lv1 = *buf1;

                                    lv = (lv1 & 0x7F) << 1;

                                    lv += (lv << 8);

                                    lv += (lv << 16);

                                    for(i = 0, j = 0; i < 8; i++, j += width_tbl[1])

                                        ((uint32_t *)cur_frm_pos)[j] = ((uint32_t *)cur_frm_pos)[j+1] = lv;

                                    LV1_CHECK(buf1,rle_v3,lv1,lp2)

                                    break;



                                default:

                                    return;

                                }

                            }



                            cur_frm_pos += 8;

                            ref_frm_pos += 8;

                        }



                        cur_frm_pos += (((width * 2) - blks_width) * 4);

                        ref_frm_pos += (((width * 2) - blks_width) * 4);

                    }

                }

                break;



            case 11:                    /********** CASE 11 **********/

                if(ref_vectors == NULL)

                    return;



                for( ; blks_height > 0; blks_height -= 8) {

                    for(lp1 = 0; lp1 < blks_width; lp1++) {

                        for(lp2 = 0; lp2 < 4; ) {

                            k = *buf1++;

                            cur_lp = ((uint32_t *)cur_frm_pos) + width_tbl[lp2 * 2];

                            ref_lp = ((uint32_t *)ref_frm_pos) + width_tbl[lp2 * 2];



                            switch(correction_type_sp[lp2 & 0x01][k]) {

                            case 0:

                                cur_lp[0] = av_le2ne32(((av_le2ne32(*ref_lp) >> 1) + correction_lp[lp2 & 0x01][k]) << 1);

                                cur_lp[width_tbl[1]] = av_le2ne32(((av_le2ne32(ref_lp[width_tbl[1]]) >> 1) + correction_lp[lp2 & 0x01][k]) << 1);

                                lp2++;

                                break;



                            case 1:

                                lv1 = (unsigned short)(correction_lp[lp2 & 0x01][*buf1++]);

                                lv2 = (unsigned short)(correction_lp[lp2 & 0x01][k]);

                                res = (unsigned short)(((av_le2ne16(((unsigned short *)ref_lp)[0]) >> 1) + lv1) << 1);

                                ((unsigned short *)cur_lp)[0] = av_le2ne16(res);

                                res = (unsigned short)(((av_le2ne16(((unsigned short *)ref_lp)[1]) >> 1) + lv2) << 1);

                                ((unsigned short *)cur_lp)[1] = av_le2ne16(res);

                                res = (unsigned short)(((av_le2ne16(((unsigned short *)ref_lp)[width_tbl[2]]) >> 1) + lv1) << 1);

                                ((unsigned short *)cur_lp)[width_tbl[2]] = av_le2ne16(res);

                                res = (unsigned short)(((av_le2ne16(((unsigned short *)ref_lp)[width_tbl[2]+1]) >> 1) + lv2) << 1);

                                ((unsigned short *)cur_lp)[width_tbl[2]+1] = av_le2ne16(res);

                                lp2++;

                                break;



                            case 2:

                                if(lp2 == 0) {

                                    for(i = 0, j = 0; i < 4; i++, j += width_tbl[1])

                                        cur_lp[j] = ref_lp[j];

                                    lp2 += 2;

                                }

                                break;



                            case 3:

                                if(lp2 < 2) {

                                    for(i = 0, j = 0; i < 6 - (lp2 * 2); i++, j += width_tbl[1])

                                        cur_lp[j] = ref_lp[j];

                                    lp2 = 3;

                                }

                                break;



                            case 8:

                                if(lp2 == 0) {

                                    RLE_V3_CHECK(buf1,rle_v1,rle_v2,rle_v3)



                                    for(i = 0, j = 0; i < 8; i++, j += width_tbl[1])

                                        cur_lp[j] = ref_lp[j];



                                    RLE_V2_CHECK(buf1,rle_v2, rle_v3,lp2)

                                    break;

                                } else {

                                    rle_v1 = 1;

                                    rle_v2 = (*buf1) - 1;

                                }

                            case 5:

                            case 7:

                                LP2_CHECK(buf1,rle_v3,lp2)

                            case 4:

                            case 6:

                                for(i = 0, j = 0; i < 8 - (lp2 * 2); i++, j += width_tbl[1])

                                    cur_lp[j] = ref_lp[j];

                                lp2 = 4;

                                break;



                            case 9:

                                av_log(s->avctx, AV_LOG_ERROR, "UNTESTED.\n");

                                lv1 = *buf1++;

                                lv = (lv1 & 0x7F) << 1;

                                lv += (lv << 8);

                                lv += (lv << 16);

                                for(i = 0, j = 0; i < 4; i++, j += width_tbl[1])

                                    cur_lp[j] = lv;

                                LV1_CHECK(buf1,rle_v3,lv1,lp2)

                                break;



                            default:

                                return;

                            }

                        }



                        cur_frm_pos += 4;

                        ref_frm_pos += 4;

                    }



                    cur_frm_pos += (((width * 2) - blks_width) * 4);

                    ref_frm_pos += (((width * 2) - blks_width) * 4);

                }

                break;



            default:

                return;

            }

        }



        for( ; strip >= strip_tbl; strip--) {

            if(strip->split_flag != 0) {

                strip->split_flag = 0;

                strip->usl7 = (strip-1)->usl7;



                if(strip->split_direction) {

                    strip->xpos += strip->width;

                    strip->width = (strip-1)->width - strip->width;

                    if(region_160_width <= strip->xpos && width < strip->width + strip->xpos)

                        strip->width = width - strip->xpos;

                } else {

                    strip->ypos += strip->height;

                    strip->height = (strip-1)->height - strip->height;

                }

                break;

            }

        }

    }

}