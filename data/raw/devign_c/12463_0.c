static int config(struct vf_instance *vf,

    int width, int height, int d_width, int d_height,

    unsigned int flags, unsigned int outfmt)

{

    switch (vf->priv->mode) {

    case 0:

    case 3:

        return ff_vf_next_config(vf,width,height*2,d_width,d_height*2,flags,outfmt);

    case 1:            /* odd frames */

    case 2:            /* even frames */

    case 4:            /* alternate frame (height-preserving) interlacing */

        return ff_vf_next_config(vf,width,height,d_width,d_height,flags,outfmt);

    }

    return 0;

}
