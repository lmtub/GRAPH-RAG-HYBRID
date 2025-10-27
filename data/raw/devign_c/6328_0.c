static double fade_gain(int curve, int64_t index, int range)

{

    double gain;



    gain = av_clipd(1.0 * index / range, 0, 1.0);



    switch (curve) {

    case QSIN:

        gain = sin(gain * M_PI / 2.0);

        break;

    case IQSIN:

        gain = 0.636943 * asin(gain);

        break;

    case ESIN:

        gain = 1.0 - cos(M_PI / 4.0 * (pow(2.0*gain - 1, 3) + 1));

        break;

    case HSIN:

        gain = (1.0 - cos(gain * M_PI)) / 2.0;

        break;

    case IHSIN:

        gain = 0.318471 * acos(1 - 2 * gain);

        break;

    case EXP:

        gain = pow(0.1, (1 - gain) * 5.0);

        break;

    case LOG:

        gain = av_clipd(0.0868589 * log(100000 * gain), 0, 1.0);

        break;

    case PAR:

        gain = 1 - sqrt(1 - gain);

        break;

    case IPAR:

        gain = (1 - (1 - gain) * (1 - gain));

        break;

    case QUA:

        gain *= gain;

        break;

    case CUB:

        gain = gain * gain * gain;

        break;

    case SQU:

        gain = sqrt(gain);

        break;

    case CBR:

        gain = cbrt(gain);

        break;

    case DESE:

        gain = gain <= 0.5 ? pow(2 * gain, 1/3.) / 2: 1 - pow(2 * (1 - gain), 1/3.) / 2;

        break;

    case DESI:

        gain = gain <= 0.5 ? pow(2 * gain, 3) / 2: 1 - pow(2 * (1 - gain), 3) / 2;

        break;

    }



    return gain;

}
