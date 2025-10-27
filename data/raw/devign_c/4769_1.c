static inline int mirror(int v, int m){

    if     (v<0) return -v;

    else if(v>m) return 2*m-v;

    else         return v;

}
