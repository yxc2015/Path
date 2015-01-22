/* The class complex defines a complex type for GPU computations.
   The template parameter value T specifies precision level */ 

#ifndef __COMPLEX_H__
#define __COMPLEX_H__

#define __CUDAC__

#ifdef __CUDAC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

#include <iostream>
#include <gqd_type.h>

using namespace std;

template <class T>
class complex
{
   public:

      DEVICE complex<T> operator=(complex<T>);
      DEVICE complex();
      DEVICE complex(double,double);
      DEVICE complex(gdd_real,gdd_real);
      DEVICE complex(gqd_real,gqd_real);
      DEVICE void init(double,double);
      void initH(double,double);
      DEVICE void init_imag();
 
      DEVICE complex operator+(complex);
      DEVICE complex operator-(complex);
      DEVICE complex operator*(complex);
      DEVICE complex adj_multiple(complex);
      DEVICE void    operator*=(complex);
      DEVICE void    operator+=(complex);
      DEVICE complex operator*(T);
      DEVICE complex operator*(int);
      DEVICE complex operator/(complex);
      DEVICE complex operator/(T);
      DEVICE void    operator/=(T);
      DEVICE T absv() {return sqrt(real*real+imag*imag);};
      DEVICE complex adj() {return complex(real,0.0-imag);};
 
      T real;
      T imag;
};

template <class T>
DEVICE complex<T> complex<T>::operator=(complex<T> a)
{
   real=a.real;
   imag=a.imag;

   return *this;
}

template <>
inline DEVICE complex<double>::complex(double a, double b)
{
   real = a;
   imag = b;
}

template <>
inline DEVICE complex<gdd_real>::complex(gdd_real a, gdd_real b)
{
   real = a;
   imag = b;
}

template <>
inline DEVICE complex<gqd_real>::complex(gqd_real a, gqd_real b)
{
   real = a;
   imag = b;
}

template <>
inline DEVICE complex<gdd_real>::complex(double a, double b)
{
   real.x = a; real.y = 0.0;
   imag.x = b; imag.y = 0.0;
}

template <>
inline DEVICE complex<gqd_real>::complex(double a, double b)
{
   real.x = a; real.y = 0.0; real.z = 0.0; real.w = 0.0;
   imag.x = b; imag.y = 0.0; imag.z = 0.0; imag.w = 0.0;
}

template <class T>
DEVICE void complex<T>::init(double a, double b)
{
   complex<T> temp(a,b);
   real = temp.real; imag = temp.imag;   
}

template <>
inline void complex<gqd_real>::initH(double a, double b)
{
   real.x = a; real.y = 0.0; real.z = 0.0; real.w = 0.0;
   imag.x = b; imag.y = 0.0; imag.z = 0.0; imag.w = 0.0; 
}

template <>
inline void complex<gdd_real>::initH(double a, double b)
{
   real.x = a; real.y = 0.0; imag.x = b; imag.y = 0.0;
}

template <>
inline void complex<double>::initH(double a, double b)
{
   real = a;  imag = b;
}

template <>
inline void complex<gqd_real>::init_imag()
{
   imag.x = 0.0; imag.y = 0.0; imag.z = 0.0; imag.w = 0.0;
}

template <>
inline void complex<gdd_real>::init_imag()
{
   imag.x = 0.0; imag.y = 0.0;
}

template <>
inline void complex<double>::init_imag()
{
   imag = 0.0;
}

template <class T>
void complex<T>::initH(double a, double b)
{
  //complex<T> temp(a,b);
  //real=temp.real; imag=temp.imag;
  //real.x=a; real.y=0.0; imag.x=b; imag.y=0.0;
}

template <class T>
DEVICE complex<T>::complex() {}

template <class T>
DEVICE complex<T> complex<T>::operator+(complex<T> a)
{
   return complex(real+a.real,imag+a.imag);
}

template <class T>
DEVICE complex<T> complex<T>::operator-(complex<T> a)
{
   return complex(real-a.real,imag-a.imag);
}

template <class T>
DEVICE complex<T> complex<T>::operator*(complex<T> a)
{
   return complex(real*a.real-imag*a.imag,imag*a.real+real*a.imag);
}

template <class T>
inline DEVICE complex<T> complex<T>::adj_multiple(complex<T> a)
{
   return complex(real*a.real+imag*a.imag,real*a.imag-imag*a.real);
}

template <class T>
DEVICE void complex<T>::operator*=(complex<T> a)
{
   T real_tmp = real;
   real = real*a.real-imag*a.imag;
   imag = imag*a.real+real_tmp*a.imag;
}


template <class T>
DEVICE void complex<T>::operator+=(complex<T> a)
{
   real = real + a.real;
   imag = imag + a.imag;
}

template <class T>
DEVICE complex<T> complex<T>::operator*(T a)
{
   return complex(real*a,imag*a);
}

template <class T>
DEVICE complex<T> complex<T>::operator*(int a)
{
   return complex(real*a,imag*a);
}

template <class T>
DEVICE complex<T> complex<T>::operator/(T a)
{
   return complex(real/a,imag/a);
}

template <class T>
DEVICE void complex<T>::operator/=(T a)
{
   real = real/a;
   imag = imag/a;
}

template <class T>
DEVICE complex<T> complex<T>::operator/(complex<T> a)
{
   return complex((real*a.real+imag*a.imag)/(a.real*a.real+a.imag*a.imag),
                  (imag*a.real-real*a.imag)/(a.real*a.real+a.imag*a.imag));
}
#endif
