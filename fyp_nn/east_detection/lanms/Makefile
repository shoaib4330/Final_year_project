CXXFLAGS = -I include  -std=c++11 -O3 $(shell /home/abdul/anaconda3/envs/py27/bin/python2.7-config --cflags)
LDFLAGS = $(shell /home/abdul/anaconda3/envs/py27/bin/python2.7-config --ldflags)

DEPS = lanms.h $(shell find include -xtype f)
CXX_SOURCES = adaptor.cpp include/clipper/clipper.cpp

LIB_SO = adaptor.so

$(LIB_SO): $(CXX_SOURCES) $(DEPS)
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $(CXX_SOURCES) --shared -fPIC

clean:
	rm -rf $(LIB_SO)
