void ff_hevcdsp_init_x86(HEVCDSPContext *c, const int bit_depth)

{

    int mm_flags = av_get_cpu_flags();



    if (bit_depth == 8) {

        if (EXTERNAL_MMX(mm_flags)) {



            if (EXTERNAL_MMXEXT(mm_flags)) {



                if (EXTERNAL_SSSE3(mm_flags) && ARCH_X86_64) {



                    EPEL_LINKS(c->put_hevc_epel, 0, 0, pel_pixels,  8);

                    EPEL_LINKS(c->put_hevc_epel, 0, 1, epel_h,      8);

                    EPEL_LINKS(c->put_hevc_epel, 1, 0, epel_v,      8);

                    EPEL_LINKS(c->put_hevc_epel, 1, 1, epel_hv,     8);



                    QPEL_LINKS(c->put_hevc_qpel, 0, 0, pel_pixels, 8);

                    QPEL_LINKS(c->put_hevc_qpel, 0, 1, qpel_h,     8);

                    QPEL_LINKS(c->put_hevc_qpel, 1, 0, qpel_v,     8);

                    QPEL_LINKS(c->put_hevc_qpel, 1, 1, qpel_hv,    8);



                }

            }

        }

    } else if (bit_depth == 10) {

        if (EXTERNAL_MMX(mm_flags)) {

            if (EXTERNAL_MMXEXT(mm_flags) && ARCH_X86_64) {



                if (EXTERNAL_SSSE3(mm_flags)) {



                    EPEL_LINKS(c->put_hevc_epel, 0, 0, pel_pixels, 10);

                    EPEL_LINKS(c->put_hevc_epel, 0, 1, epel_h,     10);

                    EPEL_LINKS(c->put_hevc_epel, 1, 0, epel_v,     10);

                    EPEL_LINKS(c->put_hevc_epel, 1, 1, epel_hv,    10);



                    QPEL_LINKS(c->put_hevc_qpel, 0, 0, pel_pixels, 10);

                    QPEL_LINKS(c->put_hevc_qpel, 0, 1, qpel_h,     10);

                    QPEL_LINKS(c->put_hevc_qpel, 1, 0, qpel_v,     10);

                    QPEL_LINKS(c->put_hevc_qpel, 1, 1, qpel_hv,    10);

                }



            }

        }

    }

}
