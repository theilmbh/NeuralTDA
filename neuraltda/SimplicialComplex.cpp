#include <list>
#include <set>
#include <vector>
#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>

#include <boost/tokenizer.hpp>

class Simplex {
public:
	std::vector<int> vertices;
	int dim;
	Simplex();
	Simplex(std::vector<int> verts);
	~Simplex();
	bool operator==(Simplex simplexB) const;
	bool operator!=(Simplex simplexB) const;
	bool operator<(Simplex simplexB) const;
	void print_vertices(std::ostream& out) const;
};

Simplex::Simplex()
{
	this->dim = -1;
}

Simplex::Simplex(std::vector<int> verts)
{	
	this->vertices.insert(this->vertices.begin(), verts.begin(), verts.end());
	this->dim = this->vertices.size()-1;
}

Simplex::~Simplex()
{

}

bool Simplex::operator==(Simplex simplexB) const
{
	return (this->vertices) == (simplexB.vertices);
}

bool Simplex::operator!=(Simplex simplexB) const
{
	return !(this->vertices == simplexB.vertices);
}

bool Simplex::operator<(Simplex simplexB) const
{
	if(this->dim != simplexB.dim)
	{
		return 0;
	}
	return this->vertices < simplexB.vertices;
}

void Simplex::print_vertices(std::ostream& out) const
{
	std::vector<int>::const_iterator it;
	it = this->vertices.begin();
	for(; it != this->vertices.end(); ++it)
	{
		out << *it << ' ';
		
	}
	out << '\n';
}

class SimplicialComplex
{
	typedef std::vector<Simplex> chain_group;
	typedef std::vector<SimplicialComplex::chain_group> scg;

	SimplicialComplex::scg simplicial_chain_groups;
public:
	SimplicialComplex(std::vector<Simplex> K);
	~SimplicialComplex();
	void add_max_simplex(std::vector<Simplex> K);
	void print_scgs();
	void save_scgs(const char* fname);
private:
	std::vector<Simplex> primary_faces(Simplex Q);
};

SimplicialComplex::SimplicialComplex(std::vector<Simplex> K)
{
	int max_dim = 0;
	std::vector<Simplex>::iterator it;
	std::vector<Simplex> *Ek;
	for(int i=0; i<K.size(); i++)
	{
		max_dim = std::max(max_dim, K[i].dim);
	}

	for(int i=0; i<max_dim+1; i++)
	{
		this->simplicial_chain_groups.push_back(std::vector<Simplex>());
	}
	add_max_simplex(K);
	for(int j=0; j<max_dim+1; j++)
	{
		Ek = &this->simplicial_chain_groups[j];
		std::sort(Ek->begin(), Ek->end());
		it = std::unique(Ek->begin(), Ek->end());
		Ek->resize(std::distance(Ek->begin(), it));
	}
}	

SimplicialComplex::~SimplicialComplex()
{

}

void SimplicialComplex::add_max_simplex(std::vector<Simplex> K2)
{
	int k;
	Simplex Q;
	std::vector<Simplex> L;
	std::vector<Simplex>::iterator it;
	std::vector<Simplex> K(K2);
	SimplicialComplex::chain_group cg;
	std::vector<Simplex> *Ek, *Ekm1;
	while(!K.empty())
	{
		Q = K.back();
		K.pop_back();
		k = Q.dim;
		if(k <= 0) { continue; }
		L = primary_faces(Q);
		Ek = &this->simplicial_chain_groups[k];
		Ekm1 = &this->simplicial_chain_groups[k-1];
		//std::cout << "L: " << L.size() << '\n';
		//std::cout << "Ek: " << Ek->size() << " Ekm1: " << Ekm1->size() << '\n';
		Ekm1->insert(Ekm1->begin(), L.begin(), L.end());
		Ek->insert(Ek->begin(), Q);

		//add_max_simplex(L);
		K.insert(K.begin(), L.begin(), L.end());
		std::sort(K.begin(), K.end());
		it = std::unique(K.begin(), K.end());
		K.resize(std::distance(K.begin(), it));


	}
}

std::vector<Simplex> SimplicialComplex::primary_faces(Simplex Q)
{
	int dim = Q.dim;

	std::vector<int> new_verts;
	std::vector<int> pts = Q.vertices;
	std::vector<Simplex> L;
	pts.insert(pts.end(), pts.begin(), pts.end()-2);

	if(dim == 0)
	{
		return L;
	}

	for(int i=0; i <= dim; i++)
	{
		new_verts.insert(new_verts.begin(), pts.begin()+i, pts.begin()+(i+dim));
		std::sort(new_verts.begin(), new_verts.end());
		Simplex new_simplex = Simplex(new_verts);
		L.push_back(new_simplex);
		new_verts.clear();	
	}
	return L;
}

void SimplicialComplex::print_scgs()
{
	int dim = 0;
	SimplicialComplex::chain_group::iterator itbeg, itend;

	for(; dim < this->simplicial_chain_groups.size(); ++dim)
	{
		std::cout << "DIMENSION: " << dim << '\n';
		std::cout << "-----------\n";
		itbeg = this->simplicial_chain_groups[dim].begin();
		itend = this->simplicial_chain_groups[dim].end();
		for(; itbeg != itend; ++itbeg)
		{
			(*itbeg).print_vertices(std::cout);
		}
		std::cout << '\n';
	}
}

void SimplicialComplex::save_scgs(const char* fname)
{
	std::ofstream scg_file(fname, std::ofstream::out);

	int dim = 0;
	SimplicialComplex::chain_group::iterator itbeg, itend;

	for(; dim < this->simplicial_chain_groups.size(); ++dim)
	{
		itbeg = this->simplicial_chain_groups[dim].begin();
		itend = this->simplicial_chain_groups[dim].end();
		for(; itbeg != itend; ++itbeg)
		{
			(*itbeg).print_vertices(scg_file);
		}
		scg_file << '\n';
	}
}

class SCGFile
{
public:
	SCGFile();
	SCGFile(const char* fname);
	~SCGFile();
	std::vector<Simplex> get_max_simplices() const;
private:
	std::vector<Simplex> max_simplices;
};

SCGFile::SCGFile(const char* fname)
{
	std::ifstream infile(fname);
	std::string line;
	std::vector<int> vertices;
	while(std::getline(infile, line))
	{
		vertices.clear();
		boost::tokenizer<boost::escaped_list_separator<char> > tk(
   		line, boost::escaped_list_separator<char>('\\', ',', '\"'));

   		for (boost::tokenizer<boost::escaped_list_separator<char> >::iterator i(tk.begin());
   			i!=tk.end();++i) 
		{
   			vertices.push_back(std::stoi(*i));
		}
		Simplex new_simplex(vertices);
		max_simplices.push_back(new_simplex);
	}
}

SCGFile::SCGFile()
{

}

SCGFile::~SCGFile()
{

}

std::vector<Simplex> SCGFile::get_max_simplices() const
{
	return max_simplices;
}


int main(int argc, char* argv[])
{
	if (argc < 3)
	{
		std::cerr << "Usage: " << argv[0] << " infile" << " outfile";
		std::cerr << std::endl;
		return 1;
	}
	SCGFile infile(argv[1]);
	SimplicialComplex sc (infile.get_max_simplices());
	sc.save_scgs(argv[2]);
	sc.print_scgs();
	return 0;
}

void simplicial_complex_test()
{
		std::vector<int> mylist;
	std::vector<int> mylist2, mylist3;
  	std::vector<int>::iterator it;

  // set some initial values:
  	for (int i=11; i<=20; ++i) mylist.push_back(i); // 1 2 3 4 5
  	for (int i=1; i<=10; ++i) mylist2.push_back(i); // 1 2 3 4 5
  	for (int i=22; i<=40; ++i) mylist3.push_back(i); // 1 2 3 4 5
	Simplex simplex1 =  Simplex(mylist);
	Simplex simplex2 = Simplex(mylist2);
	Simplex simplex3 = Simplex(mylist3);
	if(simplex1 == simplex2){
		std::cout << "True\n";
	} else {
		std::cout << "False\n";
	}

	if(simplex1 == simplex3){
		std::cout << "True\n";
	} else {
		std::cout << "False\n";
	}
	if(mylist == mylist3){
		std::cout << "True\n";
	} else {
		std::cout << "False\n";
	}
	simplex1.print_vertices(std::cout);
	simplex2.print_vertices(std::cout);
	simplex3.print_vertices(std::cout);

	std::vector<Simplex> mysc;
	mysc.push_back(simplex1);
	mysc.push_back(simplex2);
	//mysc.push_back(simplex3);
	//SimplicialComplex mysc2 (mysc);
	//mysc2.print_scgs();

	SCGFile myscgf("./scgtest.csv");
	SimplicialComplex mysc3 (myscgf.get_max_simplices());
	mysc3.print_scgs();
	mysc3.save_scgs("./scgout.scg");
}