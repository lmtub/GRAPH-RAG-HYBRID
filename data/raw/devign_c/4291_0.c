int64_t HELPER(nabs_i64)(int64_t val)

{

    if (val < 0) {

        return val;

    } else {

        return -val;

    }

}
