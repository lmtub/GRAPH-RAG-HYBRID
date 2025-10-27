static void audio_pp_nb_voices (const char *typ, int nb)

{

    switch (nb) {

    case 0:

        printf ("Does not support %s\n", typ);

        break;

    case 1:

        printf ("One %s voice\n", typ);

        break;

    case INT_MAX:

        printf ("Theoretically supports many %s voices\n", typ);

        break;

    default:

        printf ("Theoretically supports upto %d %s voices\n", nb, typ);

        break;

    }



}
