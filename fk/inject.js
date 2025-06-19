let marker = window.viewer.scene.getObjectByName("ee_marker", true);
if (!marker) { alert("No ee_marker found!"); } 
else {
    marker.userData.draggable = true;
    if (!window.dragControls) {
        window.dragControls = new THREE.DragControls([marker], window.viewer.camera, window.viewer.renderer.domElement);
        window.dragControls.addEventListener('dragend', function (event) {
            let obj = event.object;
            obj.updateMatrixWorld();
            let m = obj.matrixWorld.elements;
            let pose = [
                m[0], m[1], m[2],
                m[4], m[5], m[6],
                m[8], m[9], m[10],
                m[12], m[13], m[14]
            ];
            fetch('http://localhost:9001', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(pose)
            }).then(r=>r.text()).then(console.log);
        });
        alert("You can now drag the red marker and control the robot in Python!");
    }
}
