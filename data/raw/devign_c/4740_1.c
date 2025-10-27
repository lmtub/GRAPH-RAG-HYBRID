static void fill_picture_parameters(AVCodecContext *avctx,

                                    struct dxva_context *ctx,

                                    const struct MpegEncContext *s,

                                    DXVA_PictureParameters *pp)

{

    const Picture *current_picture = s->current_picture_ptr;

    int is_field = s->picture_structure != PICT_FRAME;



    memset(pp, 0, sizeof(*pp));

    pp->wDecodedPictureIndex         = ff_dxva2_get_surface_index(ctx, &current_picture->f);

    pp->wDeblockedPictureIndex       = 0;

    if (s->pict_type != AV_PICTURE_TYPE_I)

        pp->wForwardRefPictureIndex  = ff_dxva2_get_surface_index(ctx, &s->last_picture.f);

    else

        pp->wForwardRefPictureIndex  = 0xffff;

    if (s->pict_type == AV_PICTURE_TYPE_B)

        pp->wBackwardRefPictureIndex = ff_dxva2_get_surface_index(ctx, &s->next_picture.f);

    else

        pp->wBackwardRefPictureIndex = 0xffff;

    pp->wPicWidthInMBminus1          = s->mb_width  - 1;

    pp->wPicHeightInMBminus1         = (s->mb_height >> is_field) - 1;

    pp->bMacroblockWidthMinus1       = 15;

    pp->bMacroblockHeightMinus1      = 15;

    pp->bBlockWidthMinus1            = 7;

    pp->bBlockHeightMinus1           = 7;

    pp->bBPPminus1                   = 7;

    pp->bPicStructure                = s->picture_structure;

    pp->bSecondField                 = is_field && !s->first_field;

    pp->bPicIntra                    = s->pict_type == AV_PICTURE_TYPE_I;

    pp->bPicBackwardPrediction       = s->pict_type == AV_PICTURE_TYPE_B;

    pp->bBidirectionalAveragingMode  = 0;

    pp->bMVprecisionAndChromaRelation= 0; /* FIXME */

    pp->bChromaFormat                = s->chroma_format;

    pp->bPicScanFixed                = 1;

    pp->bPicScanMethod               = s->alternate_scan ? 1 : 0;

    pp->bPicReadbackRequests         = 0;

    pp->bRcontrol                    = 0;

    pp->bPicSpatialResid8            = 0;

    pp->bPicOverflowBlocks           = 0;

    pp->bPicExtrapolation            = 0;

    pp->bPicDeblocked                = 0;

    pp->bPicDeblockConfined          = 0;

    pp->bPic4MVallowed               = 0;

    pp->bPicOBMC                     = 0;

    pp->bPicBinPB                    = 0;

    pp->bMV_RPS                      = 0;

    pp->bReservedBits                = 0;

    pp->wBitstreamFcodes             = (s->mpeg_f_code[0][0] << 12) |

                                       (s->mpeg_f_code[0][1] <<  8) |

                                       (s->mpeg_f_code[1][0] <<  4) |

                                       (s->mpeg_f_code[1][1]      );

    pp->wBitstreamPCEelements        = (s->intra_dc_precision         << 14) |

                                       (s->picture_structure          << 12) |

                                       (s->top_field_first            << 11) |

                                       (s->frame_pred_frame_dct       << 10) |

                                       (s->concealment_motion_vectors <<  9) |

                                       (s->q_scale_type               <<  8) |

                                       (s->intra_vlc_format           <<  7) |

                                       (s->alternate_scan             <<  6) |

                                       (s->repeat_first_field         <<  5) |

                                       (s->chroma_420_type            <<  4) |

                                       (s->progressive_frame          <<  3);

    pp->bBitstreamConcealmentNeed    = 0;

    pp->bBitstreamConcealmentMethod  = 0;

}
