static int get_dimension(GetBitContext *gb, const int *dim)

{

    int t   = get_bits(gb, 3);

    int val = dim[t];

    if(val < 0)

        val = dim[get_bits1(gb) - val];

    if(!val){

        do{



            t = get_bits(gb, 8);

            val += t << 2;

        }while(t == 0xFF);

    }

    return val;

}