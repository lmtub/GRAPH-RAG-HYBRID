static double block_angle(int x, int y, int cx, int cy, MotionVector *shift)

{

    double a1, a2, diff;



    a1 = atan2(y - cy, x - cx);

    a2 = atan2(y - cy + shift->y, x - cx + shift->x);



    diff = a2 - a1;



    return (diff > M_PI)  ? diff - 2 * M_PI :

           (diff < -M_PI) ? diff + 2 * M_PI :

           diff;

}
