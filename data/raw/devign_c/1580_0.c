static int64_t truehd_layout(int chanmap)

{

    int layout = 0, i;



    for (i = 0; i < 13; i++)

        layout |= thd_layout[i] * ((chanmap >> i) & 1);



    return layout;

}
