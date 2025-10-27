static void vga_precise_update_retrace_info(VGAState *s)

{

    int htotal_chars;

    int hretr_start_char;

    int hretr_skew_chars;

    int hretr_end_char;



    int vtotal_lines;

    int vretr_start_line;

    int vretr_end_line;



    int div2, sldiv2, dots;

    int clocking_mode;

    int clock_sel;

    const int hz[] = {25175000, 28322000, 25175000, 25175000};

    int64_t chars_per_sec;

    struct vga_precise_retrace *r = &s->retrace_info.precise;



    htotal_chars = s->cr[0x00] + 5;

    hretr_start_char = s->cr[0x04];

    hretr_skew_chars = (s->cr[0x05] >> 5) & 3;

    hretr_end_char = s->cr[0x05] & 0x1f;



    vtotal_lines = (s->cr[0x06]

                    | (((s->cr[0x07] & 1) | ((s->cr[0x07] >> 4) & 2)) << 8)) + 2

        ;

    vretr_start_line = s->cr[0x10]

        | ((((s->cr[0x07] >> 2) & 1) | ((s->cr[0x07] >> 6) & 2)) << 8)

        ;

    vretr_end_line = s->cr[0x11] & 0xf;





    div2 = (s->cr[0x17] >> 2) & 1;

    sldiv2 = (s->cr[0x17] >> 3) & 1;



    clocking_mode = (s->sr[0x01] >> 3) & 1;

    clock_sel = (s->msr >> 2) & 3;

    dots = (s->msr & 1) ? 8 : 9;



    chars_per_sec = hz[clock_sel] / dots;



    htotal_chars <<= clocking_mode;



    r->total_chars = vtotal_lines * htotal_chars;

    if (r->freq) {

        r->ticks_per_char = ticks_per_sec / (r->total_chars * r->freq);

    } else {

        r->ticks_per_char = ticks_per_sec / chars_per_sec;

    }



    r->vstart = vretr_start_line;

    r->vend = r->vstart + vretr_end_line + 1;



    r->hstart = hretr_start_char + hretr_skew_chars;

    r->hend = r->hstart + hretr_end_char + 1;

    r->htotal = htotal_chars;



#if 0

    printf (

        "hz=%f\n"

        "htotal = %d\n"

        "hretr_start = %d\n"

        "hretr_skew = %d\n"

        "hretr_end = %d\n"

        "vtotal = %d\n"

        "vretr_start = %d\n"

        "vretr_end = %d\n"

        "div2 = %d sldiv2 = %d\n"

        "clocking_mode = %d\n"

        "clock_sel = %d %d\n"

        "dots = %d\n"

        "ticks/char = %lld\n"

        "\n",

        (double) ticks_per_sec / (r->ticks_per_char * r->total_chars),

        htotal_chars,

        hretr_start_char,

        hretr_skew_chars,

        hretr_end_char,

        vtotal_lines,

        vretr_start_line,

        vretr_end_line,

        div2, sldiv2,

        clocking_mode,

        clock_sel,

        hz[clock_sel],

        dots,

        r->ticks_per_char

        );

#endif

}
