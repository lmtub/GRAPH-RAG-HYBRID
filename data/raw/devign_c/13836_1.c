static int query_format(struct vf_instance *vf, unsigned int fmt)

{

        /* FIXME - really any YUV 4:2:0 input format should work */

        switch (fmt) {

        case IMGFMT_YV12:

        case IMGFMT_IYUV:

        case IMGFMT_I420:

                return ff_vf_next_query_format(vf, IMGFMT_YV12);

        }

        return 0;

}
