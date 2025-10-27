SwsVector *sws_getGaussianVec(double variance, double quality)

{

    const int length = (int)(variance * quality + 0.5) | 1;

    int i;

    double middle  = (length - 1) * 0.5;

    SwsVector *vec = sws_allocVec(length);



    if (!vec)

        return NULL;



    for (i = 0; i < length; i++) {

        double dist = i - middle;

        vec->coeff[i] = exp(-dist * dist / (2 * variance * variance)) /

                        sqrt(2 * variance * M_PI);

    }



    sws_normalizeVec(vec, 1.0);



    return vec;

}
