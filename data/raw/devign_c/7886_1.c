static int config(struct vf_instance *vf,

                  int width, int height, int d_width, int d_height,

                  unsigned int flags, unsigned int outfmt)

{

        /* FIXME - also support UYVY output? */

        return ff_vf_next_config(vf, width * vf->priv->scalew,

                              height / vf->priv->scaleh - vf->priv->skipline, d_width, d_height, flags, IMGFMT_YV12);

}
