static float ssim_end4(int sum0[5][4], int sum1[5][4], int width)

{

    float ssim = 0.0;

    int i;



    for (i = 0; i < width; i++)

        ssim += ssim_end1(sum0[i][0] + sum0[i + 1][0] + sum1[i][0] + sum1[i + 1][0],

                          sum0[i][1] + sum0[i + 1][1] + sum1[i][1] + sum1[i + 1][1],

                          sum0[i][2] + sum0[i + 1][2] + sum1[i][2] + sum1[i + 1][2],

                          sum0[i][3] + sum0[i + 1][3] + sum1[i][3] + sum1[i + 1][3]);

    return ssim;

}
