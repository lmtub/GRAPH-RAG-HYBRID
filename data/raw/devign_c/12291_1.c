static double lfo_get_value(SimpleLFO *lfo)

{

    double phs = FFMIN(100, lfo->phase / FFMIN(1.99, FFMAX(0.01, lfo->pwidth)) + lfo->offset);

    double val;



    if (phs > 1)

        phs = fmod(phs, 1.);



    switch (lfo->mode) {

    case SINE:

        val = sin(phs * 2 * M_PI);

        break;

    case TRIANGLE:

        if (phs > 0.75)

            val = (phs - 0.75) * 4 - 1;

        else if (phs > 0.25)

            val = -4 * phs + 2;

        else

            val = phs * 4;

        break;

    case SQUARE:

        val = phs < 0.5 ? -1 : +1;

        break;

    case SAWUP:

        val = phs * 2 - 1;

        break;

    case SAWDOWN:

        val = 1 - phs * 2;

        break;


    }



    return val * lfo->amount;

}