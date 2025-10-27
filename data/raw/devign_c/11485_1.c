static int vf_open(vf_instance_t *vf, char *args)

{

        vf->config=config;

        vf->query_format=query_format;

        vf->put_image=put_image;

        vf->uninit=uninit;



        vf->priv = calloc(1, sizeof (struct vf_priv_s));

        vf->priv->skipline = 0;

        vf->priv->scalew = 1;

        vf->priv->scaleh = 2;

        if (args) sscanf(args, "%d:%d:%d", &vf->priv->skipline, &vf->priv->scalew, &vf->priv->scaleh);



        return 1;

}
