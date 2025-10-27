static void init_scan_tables(H264Context *h)

{

    int i;

    for (i = 0; i < 16; i++) {

#define TRANSPOSE(x) (x >> 2) | ((x << 2) & 0xF)

        h->zigzag_scan[i] = TRANSPOSE(ff_zigzag_scan[i]);

        h->field_scan[i]  = TRANSPOSE(field_scan[i]);

#undef TRANSPOSE

    }

    for (i = 0; i < 64; i++) {

#define TRANSPOSE(x) (x >> 3) | ((x & 7) << 3)

        h->zigzag_scan8x8[i]       = TRANSPOSE(ff_zigzag_direct[i]);

        h->zigzag_scan8x8_cavlc[i] = TRANSPOSE(zigzag_scan8x8_cavlc[i]);

        h->field_scan8x8[i]        = TRANSPOSE(field_scan8x8[i]);

        h->field_scan8x8_cavlc[i]  = TRANSPOSE(field_scan8x8_cavlc[i]);

#undef TRANSPOSE

    }

    if (h->sps.transform_bypass) { // FIXME same ugly

        h->zigzag_scan_q0          = ff_zigzag_scan;

        h->zigzag_scan8x8_q0       = ff_zigzag_direct;

        h->zigzag_scan8x8_cavlc_q0 = zigzag_scan8x8_cavlc;

        h->field_scan_q0           = field_scan;

        h->field_scan8x8_q0        = field_scan8x8;

        h->field_scan8x8_cavlc_q0  = field_scan8x8_cavlc;

    } else {

        h->zigzag_scan_q0          = h->zigzag_scan;

        h->zigzag_scan8x8_q0       = h->zigzag_scan8x8;

        h->zigzag_scan8x8_cavlc_q0 = h->zigzag_scan8x8_cavlc;

        h->field_scan_q0           = h->field_scan;

        h->field_scan8x8_q0        = h->field_scan8x8;

        h->field_scan8x8_cavlc_q0  = h->field_scan8x8_cavlc;

    }

}
