// load the things we need
var express = require('express');
var app = express();

app.use(express.static('static'))

// set the view engine to ejs
app.set('view engine', 'ejs');

// use res.render to load up an ejs view file

var mnist_server = process.env.MNIST_SERVER || 'http://no-mnist-server-defined';
var http_port = 8080;

console.log('Using mnist server ' + mnist_server);

// index page 
app.get('/', function(req, res) {   
    res.render('pages/index', {
        mnist_server: mnist_server,        
    });
});

app.listen(http_port);
console.log('Listening on ' + http_port);