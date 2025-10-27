static int check_mv(H264Context *h, long b_idx, long bn_idx, int mvy_limit){

    int v;



    v = h->ref_cache[0][b_idx] != h->ref_cache[0][bn_idx] |

        h->mv_cache[0][b_idx][0] - h->mv_cache[0][bn_idx][0] + 3 >= 7U |

        FFABS( h->mv_cache[0][b_idx][1] - h->mv_cache[0][bn_idx][1] ) >= mvy_limit;



    if(h->list_count==2){

        if(!v)

            v = h->ref_cache[1][b_idx] != h->ref_cache[1][bn_idx] |

                h->mv_cache[1][b_idx][0] - h->mv_cache[1][bn_idx][0] + 3 >= 7U |

                FFABS( h->mv_cache[1][b_idx][1] - h->mv_cache[1][bn_idx][1] ) >= mvy_limit;



        if(v){

            if(h->ref_cache[0][b_idx] != h->ref_cache[1][bn_idx] |

               h->ref_cache[1][b_idx] != h->ref_cache[0][bn_idx])

                return 1;

            return

                h->mv_cache[0][b_idx][0] - h->mv_cache[1][bn_idx][0] + 3 >= 7U |

                FFABS( h->mv_cache[0][b_idx][1] - h->mv_cache[1][bn_idx][1] ) >= mvy_limit |

                h->mv_cache[1][b_idx][0] - h->mv_cache[0][bn_idx][0] + 3 >= 7U |

                FFABS( h->mv_cache[1][b_idx][1] - h->mv_cache[0][bn_idx][1] ) >= mvy_limit;

        }

    }



    return v;

}
