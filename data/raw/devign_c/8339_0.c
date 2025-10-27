void avcodec_init(void)

{

    static int inited = 0;



    if (inited != 0)

        return;

    inited = 1;



    dsputil_static_init();

}
