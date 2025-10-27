static inline void update_rice(APERice *rice, int x)

{

    rice->ksum += ((x + 1) / 2) - ((rice->ksum + 16) >> 5);



    if (rice->k == 0)

        rice->k = 1;

    else if (rice->ksum < (1 << (rice->k + 4)))

        rice->k--;

    else if (rice->ksum >= (1 << (rice->k + 5)))

        rice->k++;

}
