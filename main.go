package main

import (
	"fmt"

	"github.com/davecgh/go-spew/spew"
	"github.com/gonum/matrix"
	"github.com/gonum/matrix/mat64"
)

func pinv(base []float64, i, j int) {

	x := mat64.NewDense(2, 2, base)
	var SVD mat64.SVD
	var um, vm mat64.Dense

	SVD.Factorize(x, matrix.SVDFull)
	um.UFromSVD(&SVD)
	vm.VFromSVD(&SVD)
	w := SVD.Values(nil)

	wm := mat64.NewDense(2, 2, nil)
	leng := len(w)
	for i, v := range w {
		for j := 0; j < leng; j++ {
			if i == j && v != 0 {
				wm.Set(i, j, 1/v)
			} else {
				wm.Set(i, j, 0.0)
			}
		}
	}

}

func main() {
	elem := []float64{2, 5, 1, 3}
	x := mat64.NewDense(2, 2, elem)

	elem2 := []float64{2, 4, 1, 2}
	x2 := mat64.NewDense(2, 2, elem2)

	//elem3 := []float64{1, 0, 0, 1}
	//one := mat64.NewDense(2, 2, elem3)

	var invck mat64.Dense
	invx1 := invck.Inverse(x2)
	if invx1 != nil {
		spew.Println(invx1)
	}
	fa := mat64.Formatted(&invck, mat64.Prefix("    "), mat64.Squeeze())
	fmt.Printf("INV TEST:\na = %v\n\n", fa)
	//spew.Println("INV TEST : ", &invck)

	var m3 mat64.SVD
	m3.Factorize(x2, matrix.SVDFull)
	var um2, vm2, s2m, s3m, inv, inv2 mat64.Dense
	um2.UFromSVD(&m3)
	vm2.VFromSVD(&m3)
	w := m3.Values(nil)
	//cond := m3.Cond()
	inv.Inverse(&um2)
	s2m.Mul(&inv, x2)
	s3m.Mul(&s2m, &vm2)
	inv2.Inverse(&s3m)
	fa1 := mat64.Formatted(&um2, mat64.Prefix("      "), mat64.Excerpt(2))
	fa2 := mat64.Formatted(&vm2, mat64.Prefix("      "), mat64.Excerpt(2))
	fa3 := mat64.Formatted(&s3m, mat64.Prefix("      "), mat64.Excerpt(2))
	fa4 := mat64.Formatted(&inv2, mat64.Prefix("      "), mat64.Excerpt(2))

	m4 := mat64.NewDense(2, 2, nil)

	leng := len(w)
	for i, v := range w {
		for j := 0; j < leng; j++ {
			if i == j && v != 0 {
				m4.Set(i, j, 1/v)
			} else {
				m4.Set(i, j, 0.0)
			}
		}
	}

	fa5 := mat64.Formatted(m4, mat64.Prefix("      "), mat64.Excerpt(2))

	var mt1, mt2 mat64.Dense

	mt1.Mul(&vm2, m4)
	mt2.Mul(&mt1, &um2)

	fa6 := mat64.Formatted(&mt2, mat64.Prefix("      "), mat64.Excerpt(2))

	//spew.Println(&um2, &vm2, s2)
	//s2m.Mul(one, s2.T())
	fmt.Printf("SVD :\num2 = %v \nvm2 = %v\n\n s = %v\n p = %v\n m4 = %v\n", fa1, fa2, fa3, fa4, fa5)
	fmt.Printf("Pseudo Inverse :\nmt2 = %v \n", fa6)
	// var t1, t2, t3 mat64.Dense
	spew.Println(um2.T())

	var m1 mat64.Dense
	invx2 := m1.Inverse(x)
	if invx2 != nil {
		spew.Println(invx2)
	}
	spew.Println(&m1)

	var m2 mat64.SVD
	ok := m2.Factorize(x, matrix.SVDFull)
	if !ok {
		spew.Errorf("SVD failed")
	}
	var um, vm mat64.Dense
	um.UFromSVD(&m2)
	vm.VFromSVD(&m2)
	s := m2.Values(nil)
	spew.Println(&um, &vm, s)
}
