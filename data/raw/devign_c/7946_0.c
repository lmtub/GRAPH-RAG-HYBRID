static void shift_history(DCAEncContext *c, const int32_t *input)

{

    int k, ch;



    for (k = 0; k < 512; k++)

        for (ch = 0; ch < c->channels; ch++) {

            const int chi = c->channel_order_tab[ch];



            c->history[k][ch] = input[k * c->channels + chi];

        }

}
