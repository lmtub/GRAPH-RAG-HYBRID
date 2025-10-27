static int pad_count(const AVFilterPad *pads)

{

    int count;



    if (!pads)

        return 0;



    for(count = 0; pads->name; count ++) pads ++;

    return count;

}
