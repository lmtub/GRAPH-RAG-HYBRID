sPAPROptionVector *spapr_ovec_new(void)

{

    sPAPROptionVector *ov;



    ov = g_new0(sPAPROptionVector, 1);

    ov->bitmap = bitmap_new(OV_MAXBITS);




    return ov;

}