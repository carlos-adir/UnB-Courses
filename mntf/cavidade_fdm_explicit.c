#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static const double PI = 3.1415926535;
static const double TOLERANCE_PRESSURE = 1e-9;
static double overdx2, overdy2, over2dx, over2dy, overdtdx, overdtdy, dtover2dx, dtover2dy, mu, dt;

double top_speed(const double x)
{
    return pow(sin(PI * x), 2);
}

double initial_horizontal_speed(const double x, const double y)
{
    return pow(sin(PI * x), 2) * y * (3 * y - 2);
}

double initial_vertical_speed(const double x, const double y)
{
    return -2 * sin(PI * x) * cos(PI * x) * y * y * (y - 1);
}

void init_horizontal_speed_matrix(const int nx, const int ny, double u[nx][ny + 1])
{
    int i, j;
    double dx, dy;
    dx = 1.0 / (nx - 1);
    dy = 1.0 / (ny - 1);
    for (i = 0; i < nx; i++)
    {
        for (j = 1; j < ny; j++)
            u[i][j] = initial_horizontal_speed(i * dx, (j + 0.5) * dy);
        u[i][0] = -u[i][1];
        u[i][ny] = 2 * top_speed(i * dx) - u[i][ny - 1];
    }
}
void init_vertical_speed_matrix(const int nx, const int ny, double v[nx + 1][ny])
{
    int i, j;
    double dx, dy;
    dx = 1.0 / (nx - 1);
    dy = 1.0 / (ny - 1);
    for (j = 0; j < ny; j++)
    {
        for (i = 1; i < nx; i++)
            v[i][j] = initial_vertical_speed((i + 0.5) * dx, j * dy);
        v[0][j] = -v[1][j];
        v[nx][j] = -v[nx - 1][j];
    }
}

void print_matrix(const int nx, const int ny, const double matriz[nx][ny])
{
    int i, j;
    printf("\n");
    for (j = ny - 1; j >= 0; j--)
    {
        for (i = 0; i < nx; i++)
            printf("%c%.02f ", matriz[i][j] < 0 ? '-' : ' ', fabs(matriz[i][j]));
        printf("\n");
    }
}

void init_matrix_random_values(const int nx, const int ny, double matriz[nx][ny])
{
    int i, j;
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++)
            matriz[i][j] = rand();
}

int verify_sizes(const double dx, const double dy, const double dt, const double Re)
{
    if (dx < dt)
    {
        printf("Must have dt < dx: dx = %.1e; dt = %1.e", dx, dt);
        return 1;
    }
    if (dy < dt)
    {
        printf("Must have dt < dy: dx = %.1e; dt = %1.e", dy, dt);
        return 1;
    }
    if (1 < dx * dx * Re)
    {
        printf("The value of dx is too big! dx = %.1e > 1/sqrt(Re) = %.1e", dx, 1 / sqrt(Re));
        return 1;
    }
    if (1 < dy * dy * Re)
    {
        printf("The value of dy is too big! dy = %.1e > 1/sqrt(Re) = %.1e", dy, 1 / sqrt(Re));
        return 1;
    }
    if (0.25 * Re * dx * dx < dt)
    {
        printf("The value of dt is too big! dt = %.1e > 0.25*Re*dx^2 = %.1e", dt, 0.25 * Re * dx * dx);
        return 1;
    }
    if (0.25 * Re * dy * dy < dt)
    {
        printf("The value of dt is too big! dt = %.1e > 0.25*Re*dy^2 = %.1e", dt, 0.25 * Re * dy * dy);
        return 1;
    }
    return 0;
}

void update_ustar(const int nx, const int ny, const double u[nx][ny + 1], const double v[nx + 1][ny], double ustar[nx][ny + 1])
{
    int i, j;
    double meanv, laplace, dudx, dudy, R, dx;
    dx = 1.0 / (nx - 1);
    for (i = 1; i < nx - 1; i++)
        for (j = 1; j < ny; j++)
        {
            meanv = 0.25 * (v[i][j] + v[i - 1][j - 1] + v[i][j] + v[i - 1][j - 1]);
            dudx = over2dx * (u[i + 1][j] - u[i - 1][j]);
            dudy = over2dy * (u[i][j + 1] - u[i][j - 1]);
            laplace = overdx2 * (u[i + 1][j] - 2 * u[i][j] + u[i - 1][j]);
            laplace += overdy2 * (u[i][j + 1] - 2 * u[i][j] + u[i][j - 1]);
            R = mu * laplace - u[i][j] * dudx - meanv * dudy;
            ustar[i][j] = u[i][j] + dt * R;
        }
    for (j = 1; j < ny; j++)
    {
        ustar[0][j] = 0;
        ustar[nx - 1][j] = 0;
    }
    for (i = 0; i < nx + 1; i++)
    {
        ustar[i][0] = -ustar[i][1];
        ustar[i][ny] = 2 * top_speed(i * dx) - ustar[i][ny - 1];
    }
}

void update_vstar(const int nx, const int ny, const double u[nx][ny + 1], const double v[nx + 1][ny], double vstar[nx + 1][ny])
{
    int i, j;
    double meanu, laplace, dvdx, dvdy, R;
    for (i = 1; i < nx; i++)
        for (j = 1; j < ny - 1; j++)
        {
            meanu = 0.25 * (u[i][j] + u[i][j - 1] + u[i - 1][j] + u[i - 1][j - 1]);
            dvdx = over2dx * (v[i + 1][j] - v[i - 1][j]);
            dvdy = over2dy * (v[i][j + 1] - v[i][j - 1]);
            laplace = overdx2 * (v[i + 1][j] - 2 * v[i][j] + u[i - 1][j]);
            laplace += overdy2 * (v[i][j + 1] - 2 * v[i][j] + u[i][j - 1]);
            R = mu * laplace - v[i][j] * dvdy - meanu * dvdx;
            vstar[i][j] = v[i][j] + dt * R;
        }
    for (i = 1; i < nx; i++)
    {
        vstar[i][0] = 0;
        vstar[i][ny - 1] = 0;
    }
    for (j = 0; j < ny; j++)
    {
        vstar[0][j] = -vstar[1][j];
        vstar[nx][j] = -vstar[nx - 1][j];
    }
}

void update_force_pressure(const int nx, const int ny, const double ustar[nx][ny + 1], const double vstar[nx + 1][ny], double force_pressure[nx][ny])
{
    int i, j;
    for (i = 1; i < nx; i++)
        for (j = 1; j < ny; j++)
        {
            if (fabs(ustar[i + 1][j]) > 1)
                printf("ustar[i+1][j] = ustar[%d][%d] = %.3e\n", i + 1, j, ustar[i + 1][j]);
            if (fabs(vstar[i][j + 1]) > 1)
                printf("vstar[i][j+1] = vstar[%d][%d] = %.3e\n", i, j + 1, vstar[i][j + 1]);
            force_pressure[i][j] = overdtdx * (ustar[i + 1][j] - ustar[i][j]);
            force_pressure[i][j] += overdtdy * (vstar[i][j + 1] - vstar[i][j]);
        }
}

void update_pressure(const int nx, const int ny, const double force_pressure[nx][ny], double pressure[nx + 1][ny + 1])
{
    int i, j;
    double R, Rmax, lambda;
    do
    {
        Rmax = 0;
        for (i = 1; i < nx; i++)
        {
            for (j = 1; j < ny; j++)
            {
                R = force_pressure[i][j];
                lambda = 0;
                if (i > 1)
                {
                    R += overdx2 * (pressure[i][j] - pressure[i - 1][j]);
                    lambda -= overdx2;
                }
                if (i < nx - 1)
                {
                    R += overdx2 * (pressure[i][j] - pressure[i + 1][j]);
                    lambda -= overdx2;
                }
                if (j > 1)
                {
                    R += overdy2 * (pressure[i][j] - pressure[i][j - 1]);
                    lambda -= overdy2;
                }
                if (j < ny - 1)
                {
                    R += overdy2 * (pressure[i][j] - pressure[i][j + 1]);
                    lambda -= overdy2;
                }
                R /= lambda;
                pressure[i][j] += R;
                Rmax = fabs(R) > Rmax ? fabs(R) : Rmax;
            }
        }
    } while (Rmax > TOLERANCE_PRESSURE);
}

int main()
{
    srand(time(0));
    int nx = 4, ny = 5, nt = 10001;
    double dx, dy;
    float tmax = 3;
    double Re = 1;
    double uspeed[nx][ny + 1];
    double vspeed[nx + 1][ny];
    double pressure[nx + 1][ny + 1];
    double ustar[nx][ny + 1];
    double vstar[nx + 1][ny];
    double force_pressure[nx][ny];

    dx = 1.0 / (nx - 1);
    dy = 1.0 / (ny - 1);
    dt = tmax / (nt - 1);

    if (verify_sizes(dx, dy, dt, Re))
        return 1;

    init_matrix_random_values(nx, ny + 1, uspeed);
    init_matrix_random_values(nx + 1, ny, vspeed);
    init_matrix_random_values(nx, ny + 1, ustar);
    init_matrix_random_values(nx + 1, ny, vstar);
    init_matrix_random_values(nx + 1, ny + 1, pressure);
    init_matrix_random_values(nx, ny, force_pressure);

    init_horizontal_speed_matrix(nx, ny, uspeed);
    printf("Horizontal speed\n");
    print_matrix(nx, ny + 1, uspeed);
    init_vertical_speed_matrix(nx, ny, vspeed);
    printf("Vertical speed\n");
    print_matrix(nx + 1, ny, vspeed);

    overdx2 = 1.0 / (dx * dx);
    overdy2 = 1.0 / (dy * dy);
    over2dx = 1.0 / (2 * dx);
    over2dy = 1.0 / (2 * dy);
    overdtdx = 1.0 / (dt * dx);
    overdtdy = 1.0 / (dt * dy);
    dtover2dx = dt / (2 * dx);
    dtover2dy = dt / (2 * dy);
    mu = 1.0 / Re;

    /*for (k = 1; k < nt; k++)
{
}*/
    printf("Got here! Update Ustar!\n");
    update_ustar(nx, ny, uspeed, vspeed, ustar);
    printf("Printing U star\n");
    print_matrix(nx, ny + 1, ustar);
    printf("Got here! Update Vstar!\n");
    update_vstar(nx, ny, uspeed, vspeed, vstar);
    printf("Printing V star\n");
    print_matrix(nx + 1, ny, vstar);
    printf("Update force pressure\n");
    update_force_pressure(nx, ny, ustar, vstar, force_pressure);
    printf("Printing force pressure\n");
    print_matrix(nx, ny, force_pressure);
    printf("Update pressure\n");
    /* update_pressure(nx, ny, force_pressure, pressure); */
    printf("Printing pressure\n");
    print_matrix(nx, ny, force_pressure);
    return 0;
}