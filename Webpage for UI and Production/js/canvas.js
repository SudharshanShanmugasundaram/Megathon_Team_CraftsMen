var canvas = document.querySelector('canvas');

canvas.width=window.innerWidth;
canvas.height=window.innerHeight;
var c = canvas.getContext('2d');
console.log(canvas);
var mouse = {
	x :undefined,
	y: undefined
}

var maxRadius = 50;
var minRadius = 5;

window.addEventListener('mousemove', function(event) {
	mouse.x = event.x;
	mouse.y = event.y;
})

window.addEventListener('resize', function() {
	canvas.width = window.innerWidth;
	canvas.height = window.innerHeight;
})

var color = [
	'#9ABC96',
	'#807F82',
	'#F3F3F4',
	'#C0CADD',
	'#191D4B',
]
function Circle(x,y,dx,dy,radius) {
	this.x=x;
	this.y=y;
	this.dx=dx;
	this.dy=dy;
	this.radius = radius;
	this.i = Math.floor(Math.random() *  4 );
	this.minRadius = radius;

	this.draw = function() {

		
		c.beginPath();
		c.arc(this.x, this.y, this.radius, 0, Math.PI*2, false);
		//c.strokeStyle = 'rgb(' + this.r + ',' + this.g + ',' + this.b + ')';
		c.fillStyle = color[this.i];
		//c.stroke();
		c.fill();

	}

	this.update = function() {

		if(this.x+this.radius>=innerWidth || this.x-this.radius<=0)
		{
			this.dx=-this.dx;
			
		}
		if(this.y+this.radius>=innerHeight || this.y-this.radius<=0)
		{
			
			this.dy=-this.dy;

		}
		this.y = this.y + this.dy;
		this.x = this.x + this.dx;
		
		if(mouse.x - this.x < 50 && mouse.x - this.x > -50 && mouse.y - this.y < 50 && mouse.y - this.y > -50) {
			if(this.radius < maxRadius)
				this.radius++;
		}
		else {
			if(this.radius<this.minRadius)
				this.radius = this.minRadius;
			else
				this.radius --;
		}

		
		this.draw();

		
	}
}

var circleArray = [];

for (var i = 0; i < 500; ++i) {

	var radius = Math.random() * 9 + 1;
	var x = Math.random() * (window.innerWidth - radius * 2 ) + radius;
	var y = Math.random() * (window.innerHeight - radius * 2 ) + radius;
	var dx = (Math.random() - 0.5)*2;
	var dy = (Math.random() - 0.5)*2;

	circleArray.push(new Circle(x,y,dx,dy,radius));

}

function animate() {

	requestAnimationFrame(animate);
	c.clearRect(0,0,innerWidth,innerHeight);
				
	//circle.update();
	for (var i = 0; i < circleArray.length; ++i) {
		circleArray[i].update();	
	}
	
}
animate();
