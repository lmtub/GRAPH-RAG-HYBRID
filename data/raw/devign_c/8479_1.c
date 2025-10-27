static int put_image(struct vf_instance *vf, mp_image_t *mpi, double pts)

{

        mp_image_t *dmpi;



        // hope we'll get DR buffer:

        dmpi=ff_vf_get_image(vf->next, IMGFMT_YV12,

                          MP_IMGTYPE_TEMP, MP_IMGFLAG_ACCEPT_STRIDE |

                          ((vf->priv->scaleh == 1) ? MP_IMGFLAG_READABLE : 0),

                          mpi->w * vf->priv->scalew,

                          mpi->h / vf->priv->scaleh - vf->priv->skipline);



        toright(dmpi->planes, mpi->planes, dmpi->stride,

                mpi->stride, mpi->w, mpi->h, vf->priv);



        return ff_vf_next_put_image(vf,dmpi, pts);

}
