const fs=require('fs'),vm=require('vm'),assert=require('assert');
const path=require('path').join(__dirname,'index.html');
const source=fs.readFileSync(path,'utf8').match(/<script>([\s\S]*?)<\/script>/)[1];
const c={};vm.createContext(c);vm.runInContext(source+';globalThis.api={numbers,booking,trailing,subnet,solve,filterDemand}',c);const a=c.api;let cases=0;
function test(name,fn){fn();cases++;console.log('PASS '+name)}
function near(x,y){assert(Math.abs(x-y)<1e-10,`${x} != ${y}`)}
test('IPv4 /24 known boundary',()=>{let r=a.subnet('192.168.10.42/24');assert.equal(r.network,'192.168.10.0');assert.equal(r.broadcast,'192.168.10.255');assert.equal(r.size,256);assert.equal(r.usable,254)});
test('IPv4 /0 full range',()=>{let r=a.subnet('192.168.10.42/0');assert.equal(r.network,'0.0.0.0');assert.equal(r.broadcast,'255.255.255.255');assert.equal(r.size,4294967296);assert.equal(r.usable,4294967294)});
test('IPv4 /31 and /32 conventions',()=>{assert.equal(a.subnet('10.0.0.1/31').usable,2);assert.equal(a.subnet('10.0.0.1/32').usable,1)});
test('IPv4 invalid input rejected',()=>{for(let s of ['','999.1.1.1/24','1.1.1.1/33','1.1.1/24','1.1.1.1/-1'])assert.throws(()=>a.subnet(s))});
test('Linear system known solution and residual',()=>{let r=a.solve('2,1,-1,8\n-3,-1,2,-11\n-2,1,2,-3');r.x.forEach((v,i)=>near(v,[2,3,-1][i]));near(r.residual,0)});
test('Linear system uses pivoting',()=>{let r=a.solve('0,1,0,2\n1,0,0,1\n0,0,1,3');r.x.forEach((v,i)=>near(v,[1,2,3][i]))});
test('Singular and malformed systems rejected',()=>{for(let s of ['1,1,1,3\n2,2,2,6\n3,3,3,9','1,2,3,4','1,2,,4\n2,3,4,5\n3,4,5,6','1,2,Infinity,4\n2,3,4,5\n3,4,5,6'])assert.throws(()=>a.solve(s))});
test('Booking threshold and flags',()=>{let r=a.booking([1,2,3],1);near(r.mean,2);near(r.sd,Math.sqrt(2/3));assert.equal(r.flags.join(','),'3');assert.equal(a.booking([2,2],1).flags.length,0)});
test('Invalid booking count and threshold',()=>{assert.throws(()=>a.booking([-1,2],1));assert.throws(()=>a.booking([1,2],6));assert.throws(()=>a.booking([1,2],NaN))});
test('Time-series leakage-free known forecasts',()=>{let r=a.trailing([1,2,3,4],2);assert.equal(r.predictions.join(','),'1.5,2.5');near(r.mae,1.5);near(r.next,3.5)});
test('Invalid forecast windows rejected',()=>{for(let w of [0,4,1.5,NaN])assert.throws(()=>a.trailing([1,2,3,4],w))});
test('Observation parsing rejects empty/nonfinite/excessive input',()=>{for(let s of ['', '1,,2','1,Infinity','1,NaN',Array(201).fill('1').join(',')])assert.throws(()=>a.numbers(s));assert.equal(a.numbers('0, 2,3').join(','),'0,2,3')});
test('Demand filter reproducible aggregates',()=>{let r=a.filterDemand('all','all');assert.equal(r.rows.length,8);near(r.mean,49.875);r=a.filterDemand('clear','weekday');assert.equal(r.rows.length,3);near(r.mean,52)});
console.log(`${cases} deterministic browser-calculation checks passed.`);
