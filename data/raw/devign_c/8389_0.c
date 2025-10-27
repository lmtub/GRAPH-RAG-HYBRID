SwsVector *sws_cloneVec(SwsVector *a)

{

    int i;

    SwsVector *vec = sws_allocVec(a->length);



    if (!vec)

        return NULL;



    for (i = 0; i < a->length; i++)

        vec->coeff[i] = a->coeff[i];



    return vec;

}
